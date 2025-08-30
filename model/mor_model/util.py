from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import math
import torch
import torch.nn as nn

from transformers.utils import ModelOutput


class LinearRouter(nn.Module):
    def __init__(self, config, out_dim=1):
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.hidden_size, out_dim, bias=False)
        self.router.weight.data.normal_(mean=0.0, std=config.initializer_range)
        
    def forward(self, x):
        return self.router(x)


class MLPRouter(nn.Module):
    def __init__(self, config, out_dim=1):
        super().__init__()
        self.config = config
        self.router = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2, bias=False),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, out_dim, bias=False)
        )
        for layer in self.router:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=config.initializer_range)
    
    def forward(self, x):
        return self.router(x)


class WideMLPRouter(nn.Module):
    def __init__(self, config, out_dim=1):
        super().__init__()
        self.config = config
        self.router = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2, bias=False),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, out_dim, bias=False)
        )
        for layer in self.router:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=config.initializer_range)
    
    def forward(self, x):
        return self.router(x)


class DepthAwareRouter(nn.Module):
    """
    Depth-aware Router:
      logits = Linear(h) + alpha * Adapter(e_l)
    - Shared backbone: Linear(h) -> [*, out_dim]
    - Depth-specific adapter: MLP(e_l) -> [out_dim], broadcast to match logits
    - Recursion embeddings: learnable, sinusoidal-initialized
    """

    def __init__(self, config, num_recursion: int, out_dim: int = 1):
        super().__init__()
        self.config = config
        d = config.hidden_size
        self.out_dim = out_dim

        # Shared router backbone
        self.router = nn.Linear(d, out_dim, bias=False)
        self.router.weight.data.normal_(mean=0.0, std=config.initializer_range)

        # Adapter width
        ratio = 0.25
        d_mid = max(1, int(d * ratio))

        # Depth-specific adapter (small MLP)
        self.adapter = nn.Sequential(
            nn.Linear(d, d_mid, bias=True),
            nn.GELU(),
            nn.Linear(d_mid, out_dim, bias=True),
        )
        for layer in self.adapter:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Alpha scaling
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self.num_recursion = num_recursion
        self.recur_embeds = nn.Parameter(torch.zeros(self.num_recursion, d), requires_grad=True)
        self._init_recur_embeddings(self.recur_embeds)

    @torch.no_grad()
    def _init_recur_embeddings(self, embeds: torch.Tensor):
        # Sinusoidal positional encoding style init
        # embeds: [L, D]
        L, D = embeds.shape
        position = torch.arange(L, dtype=torch.float32, device=embeds.device).unsqueeze(1)  # [L, 1]
        div_term = torch.exp(torch.arange(0, D, 2, dtype=torch.float32, device=embeds.device) * (-math.log(10000.0) / D))
        pe = torch.zeros(L, D, device=embeds.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        embeds.copy_(pe)

    def forward(self, x: torch.Tensor, recur_idx: int):
        """
        x: [..., hidden_size]
        Returns: logits of shape [..., out_dim]
        """
        # Shared backbone logits
        z = self.router(x)  # [..., out_dim]

        # Depth embedding -> adapter bias
        e_vec = self.recur_embeds[recur_idx]  # [D]
        bias_vec = self.adapter(e_vec)  # [out_dim]

        # Broadcast add
        while bias_vec.dim() < z.dim():
            bias_vec = bias_vec.unsqueeze(0)
        logits = z + self.alpha * bias_vec  # [..., out_dim]
        return logits


ROUTER_TYPES = {
    "linear": LinearRouter, 
    "mlp": MLPRouter, 
    "wide_mlp": WideMLPRouter,
    "depth_aware": DepthAwareRouter,
}


@dataclass
class MoRLayerOutputWithPast(ModelOutput):

    hidden_state: Optional[torch.FloatTensor] = None
    attention_weights: Optional[torch.FloatTensor] = None
    selected_tokens: Optional[torch.FloatTensor] = None
    sampling_loss: Optional[torch.FloatTensor] = None
    sampling_acc: Optional[torch.FloatTensor] = None
    sampling_topk_acc: Optional[torch.FloatTensor] = None
    uniformity: Optional[torch.FloatTensor] = None
    dead_token_seq: Optional[torch.FloatTensor] = None
    balancing_loss: Optional[torch.FloatTensor] = None
    balancing_ratio: Optional[torch.FloatTensor] = None
    router_z_loss: Optional[torch.FloatTensor] = None