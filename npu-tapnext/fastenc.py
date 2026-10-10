"""torch re-implementation of aie.utils.bfp.encode(rounding="conv_even"): shared exponent = the block's max biased
fp32 exponent, mantissa = rint(signed 24-bit mantissa / 2^(16 + shift)) clipped to [-128, 127] (saturates, no
exponent bump), far blocks (shift >= 32) -> -1 / 0."""
import time
import numpy as np
import torch


def encode_rows_torch(x: torch.Tensor) -> np.ndarray:
    R, K = x.shape
    bits = x.detach().float().contiguous().view(torch.int32).reshape(R, K // 8, 8)
    sign = bits < 0
    exp = (bits >> 23) & 0xFF
    mant = (bits & 0x7FFFFF) | torch.where(exp != 0, 0x800000, 0)
    max_exp = exp.amax(-1, keepdim=True)
    shift = (max_exp - exp).to(torch.int64)
    far = shift >= 32
    signed = torch.where(sign, -mant.to(torch.int64), mant.to(torch.int64)).double()
    v = torch.round(signed / torch.exp2(torch.clamp(16 + shift, max=62).double()))
    v = v.clamp(-128, 127)
    v = torch.where(far, torch.where(sign, -1.0, 0.0), v).to(torch.int8).view(torch.uint8)
    out = torch.cat([max_exp[..., :1].to(torch.uint8), v], dim=-1)   # [R, K/8, 9]
    return out.numpy()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/opt/xilinx/xrt/python")
    from aie.utils import bfp
    torch.manual_seed(0)
    for scale in (3.0, 0.01, 300.0):
        x = torch.randn(1024, 768) * scale
        x[0, :16] = 0; x[1, 8:16] = 1e-30
        t0 = time.perf_counter(); a = encode_rows_torch(x); t1 = time.perf_counter()
        b = bfp.encode(x.numpy().astype(np.float32), rounding="conv_even").reshape(1024, 96, 9); t2 = time.perf_counter()
        print(f"scale {scale}: equal {np.array_equal(a, b)} torch {1e3*(t1-t0):.1f} ms numpy {1e3*(t2-t1):.1f} ms")
