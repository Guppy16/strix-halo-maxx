import torch, numpy as np
from safetensors.torch import load_file
sd = load_file("/work/data/models/tapnextpp_256.safetensors")
for k, v in sd.items():
    if "blocks.0." in k or "blocks" not in k:
        print(f"{k:70s} {str(tuple(v.shape)):24s} {v.dtype} min {v.float().min():.4f} max {v.float().max():.4f} mean {v.float().mean():.4f}")
for i in range(12):
    ap = sd[f"blocks.{i}.ssm_block.recurrent_block.rg_lru.a_param"].float()
    sp = torch.nn.functional.softplus(ap)
    print(f"block {i}: a_param [{ap.min():.3f},{ap.max():.3f}] softplus [{sp.min():.4f},{sp.max():.4f}] -> log_a min {-8*sp.max():.3f}")
