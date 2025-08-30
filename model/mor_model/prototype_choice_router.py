import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from types import SimpleNamespace

from transformers import LlamaConfig
from model.mor_model.util import MoRLayerOutputWithPast
from model.kv_caches.cache_utils import Cache
from router import TokenDistributionRouter 

# class LPRStylePrototypeRouter(nn.Module):
#     def __init__(self, config: LlamaConfig, num_prototypes: int, lpr_config: SimpleNamespace):
#         super().__init__()
#         self.hidden_size = config.hidden_size
#         self.num_prototypes = num_prototypes
#         self.lpr_config = lpr_config
#         self.latent_dim = lpr_config.latent_dim

#         self.encoder_norm = nn.LayerNorm(self.hidden_size)
#         self.encoder_act = nn.SiLU()
#         self.encoder_mean = nn.Linear(self.hidden_size, self.latent_dim)
#         self.encoder_log_var = nn.Linear(self.hidden_size, self.latent_dim)
        
#         prototypes_raw = torch.randn(num_prototypes, self.latent_dim)
#         prototypes_normalized = F.normalize(prototypes_raw, p=2, dim=1)
#         self.prototypes = nn.Parameter(prototypes_normalized)

#         self.beta_kl = lpr_config.beta_kl
#         self.beta_div = lpr_config.beta_div
#         self.beta_align = lpr_config.beta_align
#         self.ema_momentum = lpr_config.ema_momentum

#     @torch.no_grad()
#     def _update_prototypes_ema(self, latent_z_flat, chosen_indices_flat):
#         if not self.training:
#             return
#         unique_indices = torch.unique(chosen_indices_flat)
#         for idx in unique_indices:
#             mask = (chosen_indices_flat == idx)
#             if mask.any():
#                 batch_mean = latent_z_flat[mask].mean(dim=0)
#                 old_prototype = self.prototypes.data[idx]
#                 new_prototype = self.ema_momentum * old_prototype + (1.0 - self.ema_momentum) * batch_mean
#                 self.prototypes.data[idx] = F.normalize(new_prototype, p=2, dim=0)

#     def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
#         batch_size, seq_len, _ = hidden_states.shape
#         flat_hidden_states = hidden_states.view(-1, self.hidden_size)

#         norm_states = self.encoder_norm(flat_hidden_states)
#         act_states = self.encoder_act(norm_states)
#         mean = self.encoder_mean(act_states)
#         log_var = self.encoder_log_var(act_states)
#         std = torch.exp(0.5 * log_var)
#         epsilon = torch.randn_like(std)
#         latent_z_flat = mean + epsilon * std

#         loss_kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1).mean()

#         proto_norm = F.normalize(self.prototypes, p=2, dim=1)
#         ortho_matrix = torch.matmul(proto_norm, proto_norm.t())
#         identity = torch.eye(self.num_prototypes, device=ortho_matrix.device)
#         loss_div = torch.norm(ortho_matrix - identity, p='fro')

#         scores = torch.matmul(latent_z_flat, proto_norm.t())
        
#         soft_probs = F.softmax(scores, dim=-1)
#         aggregated_prototypes = torch.matmul(soft_probs, proto_norm)
#         loss_align = F.mse_loss(aggregated_prototypes, latent_z_flat.detach())

#         chosen_indices_flat = torch.argmax(scores, dim=-1)
        
#         # EMA更新现在依赖于正确的 .training 状态
#         self._update_prototypes_ema(latent_z_flat, chosen_indices_flat)

#         return {
#             "chosen_indices": chosen_indices_flat.view(batch_size, seq_len),
#             "soft_probs": soft_probs.view(batch_size, seq_len, -1),
#             "loss_kl": loss_kl * self.beta_kl,
#             "loss_div": loss_div * self.beta_div,
#             "loss_align": loss_align * self.beta_align,
#         }

class PrototypeMoRLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, all_shared_blocks: nn.ModuleList, num_recursion: int, mor_config: SimpleNamespace):
        super().__init__()
        self.config = config
        self.all_shared_blocks = all_shared_blocks
        self.num_recursion = num_recursion
        self.mor_config = mor_config

        lpr_cfg_copy = mor_config.lpr
        lpr_cfg_copy.num_experts = 2**num_recursion
        lpr_cfg_copy.num_experts_per_tok = 1
        lpr_cfg_copy.hidden_size = config.hidden_size
        lpr_cfg_copy.model_type = config.model_type

        self.router = TokenDistributionRouter(
            config=lpr_cfg_copy,
            layer_id=0
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None, 
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> MoRLayerOutputWithPast:
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat_hidden_states = hidden_states.view(-1, hidden_dim)

        routing_weights, total_lpr_loss, topk_idx, scores, _ = self.router(flat_hidden_states)
        
        chosen_prototype_indices = topk_idx.view(batch_size, seq_len)
        
        final_hidden_states = hidden_states

        present_key_values = [] if use_cache else None

        for depth_idx in range(self.num_recursion):
            active_mask_2d = ((chosen_prototype_indices >> depth_idx) & 1) == 1
            active_indices_tuple = torch.where(active_mask_2d)
            num_active_tokens = active_indices_tuple[0].numel()
            
            current_shared_block = self.all_shared_blocks[depth_idx]

            if num_active_tokens == 0:
                if use_cache:
                    if past_key_value is not None:
                         start_idx = depth_idx * len(current_shared_block)
                         for i in range(len(current_shared_block)):
                            present_key_values.append(past_key_value[start_idx + i])
                    else:
                        for _ in current_shared_block:
                            present_key_values.append(None)
                continue
            
            # 步骤 1: GATHER
            active_hidden_states = final_hidden_states[active_indices_tuple]

            # 应用深度原型偏置
            depth_prototype_index = 1 << depth_idx
            depth_bias = self.router.expert_keys[depth_prototype_index]
            biased_input = active_hidden_states + depth_bias.to(active_hidden_states.dtype)

            # 步骤 2: PREPARE INPUTS - 为新的密集批次准备输入
            
            # a) 准备 Attention Mask
            sparse_attention_mask = None
            if attention_mask is not None:
                bs, _, q_len, k_len = attention_mask.shape
                
                row_indices = active_indices_tuple[1].view(1, 1, num_active_tokens, 1).expand(-1, -1, -1, k_len)
                mask_rows_selected = torch.gather(attention_mask[active_indices_tuple[0]], 2, row_indices)
                col_indices = active_indices_tuple[1].view(1, 1, 1, num_active_tokens).expand(-1, -1, num_active_tokens, -1)
                sparse_attention_mask = torch.gather(mask_rows_selected, 3, col_indices)

            # b) 准备 Position IDs
            sparse_position_ids = position_ids[active_indices_tuple] if position_ids is not None else None
            
            # 步骤 3: PROCESS
            temp_states = biased_input
            for i, layer in enumerate(current_shared_block):
                layer_outputs = layer(
                    temp_states,
                    attention_mask=sparse_attention_mask,
                    position_ids=sparse_position_ids,
                    past_key_value=past_key_value, 
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **kwargs,
                )
                temp_states = layer_outputs[0]
                if use_cache:
                    pass

            # 步骤 4: SCATTER
            
            # a) 获取权重，来源是完整的scores的softmax
            soft_probs_flat = F.softmax(scores, dim=-1)
            chosen_indices_flat = chosen_prototype_indices.view(-1)
            weights_flat = soft_probs_flat[torch.arange(soft_probs_flat.size(0), device=soft_probs_flat.device), chosen_indices_flat].unsqueeze(1)
            active_weights = weights_flat[active_mask_2d.view(-1)].to(temp_states.dtype)
            
            # b) 应用权重
            src_weighted = temp_states * active_weights

            # c) & d) 执行 scatter_add 模式的写回
            scatter_buffer = torch.zeros_like(final_hidden_states)
            scatter_buffer.index_put_(active_indices_tuple, src_weighted)
            final_hidden_states = final_hidden_states + scatter_buffer

        return MoRLayerOutputWithPast(
            hidden_state=final_hidden_states,
            past_key_value=past_key_value,
            balancing_loss=total_lpr_loss,
            router_z_loss=None, attention_weights=None, selected_tokens=None, 
            sampling_loss=None, sampling_acc=None, sampling_topk_acc=None, 
            uniformity=None, dead_token_seq=None, balancing_ratio=None,
        )


    def update_expert_keys(self):
        self.router.update_expert_keys()