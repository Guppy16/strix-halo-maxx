# numpy model of nk8 recip1p_i: bit-trick seed (bf16), bf16 eps, fp32 r, 3 Newton corrections
import numpy as np
from ml_dtypes import bfloat16
bf = lambda x: np.asarray(x, np.float32).astype(bfloat16).astype(np.float32)
d = np.float32(1) + np.exp(np.linspace(-20, 40, 200001)).astype(np.float32)
dh = bf(d); dl = bf(d - dh)
r0 = (np.int16(0x7EF3) - dh.astype(bfloat16).view(np.int16)).view(bfloat16).astype(np.float32)
eps = np.float32(1) - dh * r0 - dl * r0
r = r0 + r0 * bf(eps)
for _ in range(1):
    rh = bf(r); rl = bf(r - rh)
    e2 = np.float32(1) - dh * rh - dh * rl - dl * rh
    r = r + rh * bf(e2)
print("seed max rel err", np.max(np.abs(r0 * d - 1)), "final max rel", np.abs(r * d - 1).max(),
      "bf16 flips vs exact", np.mean(bf(r) != bf(1 / d.astype(np.float64))))
