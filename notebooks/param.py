import torch

state_dict = torch.load("pytorch_model.bin", map_location="cpu")
print(state_dict.keys())

param_name = "model.layers.2.mor_router.recur_embeds"

if param_name in state_dict:
    tensor = state_dict[param_name]
    print(tensor.shape)
    print(tensor)
else:
    print(f"{param_name} 不在 state_dict 里")
