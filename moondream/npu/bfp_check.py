import sys, types
import numpy as np, torch
sys.modules["md2fast"] = types.SimpleNamespace(Lin=object)
sys.path.insert(0, "/mo")
from bfpemu import bfp_quantize
from aie.utils import bfp
rng = np.random.default_rng(0)
x = (rng.standard_normal((512, 1152)) * np.exp2(rng.integers(-12, 12, (512, 1152)))).astype(np.float32)
x[:5] = 0; x[5, :8] = 1.999; x[6, :8] = -2.0; x[7] = rng.standard_normal(1152).astype(np.float32) * 1e-3
ref = bfp.quantize(x, rounding="conv_even")
ours = bfp_quantize(torch.from_numpy(x)).numpy()
print("bit-exact:", np.array_equal(ref.view(np.uint32), ours.view(np.uint32)), "mismatches:", int((ref != ours).sum()), "of", ref.size)
rel = np.linalg.norm(ours - x) / np.linalg.norm(x); print("rms rel err vs fp32 (wild-range data):", rel)
g = rng.standard_normal((729, 1152)).astype(np.float32)
print("rms rel err, gaussian rows:", np.linalg.norm(bfp_quantize(torch.from_numpy(g)).numpy() - g) / np.linalg.norm(g))
