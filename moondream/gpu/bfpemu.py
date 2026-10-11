"""The NPU's bfp16ebs8 matmul, emulated on the GPU, to measure its accuracy cost on the 493 frames.

bfp16ebs8 (AIE2P): blocks of 8 values along K share one exponent; each value keeps an 8-bit
two's-complement mantissa. `bfp_quantize` matches mlir-aie's `aie.utils.bfp.quantize(rounding=
"conv_even")` (round to nearest even; a mantissa that rounds to +128 raises the block's exponent).
The mlir-aie GEMM also rounds its accumulator to bf16 every 64 along K, which BfpLin reproduces.

    venv-rocm714/bin/python bench_bfp.py --vlin BfpLin --nograph
"""

import torch
import torch.nn.functional as F

from md2fast import Lin

BLOCK = 8
ACC_K = 64  # the NPU GEMM stores C as bf16 after every 64 of K


def bfp_quantize(x: torch.Tensor) -> torch.Tensor:
    """What the NPU reads of x (..., K), K % 8 == 0, as float32."""
    b = x.float().reshape(*x.shape[:-1], -1, BLOCK)
    _, e = torch.frexp(b.abs().amax(-1, keepdim=True))  # amax in [2^(e-1), 2^e)
    scale = torch.ldexp(torch.ones_like(b[..., :1]), e - 7)  # 2^(biased exp - 127 - 6)
    m = torch.round(b / scale)
    scale = torch.where(m.amax(-1, keepdim=True) > 127, scale * 2, scale)
    m = torch.round(b / scale).clamp(-128, 127)
    return (m * scale).reshape(x.shape)


def hadamard(n: int, device) -> torch.Tensor:
    """Orthonormal Sylvester Hadamard matrix, n a power of two."""
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / n**0.5


class BfpLin(Lin):
    """Both operands in bfp16ebs8, bf16 accumulator every 64 of K: the mlir-aie fast GEMM."""

    quant_x = True
    quant_w = True
    acc_bf16 = True
    rotate = 0  # >0: Hadamard rotation of this size along K, for both operands (cancels in x @ w.T)

    def __init__(self, weight, bias, dtype):
        super().__init__(weight, bias, dtype)
        self.pad = -weight.shape[1] % max(BLOCK, self.rotate)
        w = F.pad(self.w.float(), (0, self.pad))
        if self.rotate:
            self.h = hadamard(self.rotate, w.device)
            w = self._rot(w)
        self.wq = bfp_quantize(w) if self.quant_w else w

    def _rot(self, t):
        return (t.reshape(*t.shape[:-1], -1, self.rotate) @ self.h).reshape(t.shape)

    def __call__(self, x):
        x = F.pad(x.float(), (0, self.pad))
        if self.rotate:
            x = self._rot(x)
        if self.quant_x:
            x = bfp_quantize(x)
        c = None
        for k in range(0, x.shape[-1], ACC_K):
            p = x[..., k : k + ACC_K] @ self.wq[:, k : k + ACC_K].t()
            c = p if c is None else c + p
            if self.acc_bf16:
                c = c.to(torch.bfloat16).float()
        if self.b is not None:
            c = c + self.b.float()
        return c.to(self.w.dtype)


class BfpWLin(BfpLin):
    """Only the weights in bfp16ebs8 (activations stay bf16): how much each operand costs."""

    quant_x = False


class BfpWAcc32Lin(BfpWLin):
    """Weights in bfp16ebs8, fp32 accumulator: is it the weights or the bf16 accumulator?"""

    acc_bf16 = False


class AccLin(BfpLin):
    """No block rounding at all, only the bf16 accumulator every 64 of K."""

    quant_x = False
    quant_w = False


class BfpAcc32Lin(BfpLin):
    """The fixed kernel: both operands bfp16ebs8, fp32 accumulator over the whole of K."""

    acc_bf16 = False


class BfpRotLin(BfpAcc32Lin):
    """Same, with a 64-wide Hadamard rotation along K evening out magnitudes inside each block."""

    rotate = 64
