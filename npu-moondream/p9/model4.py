# offline error models for the saved v4b output (no NPU): which numerics step explains the NPU-vs-model gap?
import sys
import numpy as np
from aie.utils import bfp
from ml_dtypes import bfloat16
d = np.load(sys.argv[1]); o, ref, Q, K, V, Vp, scale, NK = (d[k] for k in ("o", "ref", "Q", "K", "V", "Vp", "scale", "NK")); NK = int(NK)
bf = lambda x: np.asarray(x, np.float32).astype(bfloat16).astype(np.float32)
def rq(m, rounding="conv_even"):
    f = np.asarray(m, np.float32); f = f.reshape(-1, 8) if f.shape[-1] % 8 == 0 else np.concatenate([f, np.zeros((f.shape[0], (-f.shape[1]) % 8), np.float32)], axis=1).reshape(-1, 8)
    return bfp.decode(bfp.encode(f, rounding=rounding)).reshape(m.shape[0], -1)[:, :m.shape[1]]
def exp_srs_np(x, sym=True):
    xh = bf(x); xl = bf(x - xh)
    L = np.float32(1.4426950408889634); LH = bf(np.array([L], np.float32))[0]; LL = np.float32(L - LH)
    za = (xh * LH + xh * LL + xl * LH).astype(np.float32)
    qz = np.clip(np.rint(za * 256), -60 * 256, 60 * 256)
    n = np.floor((qz + 128) / 256) if sym else np.floor(qz / 256); fr = qz - n * 256; f = (fr / 256).astype(np.float32)
    p = bf(f * bf(np.array([0.05550411], np.float32))[0])
    p = bf(np.float32(0.24022651) + f * p); p = bf(np.float32(0.69314718) + f * p); p = (np.float32(1.0) + f * p).astype(np.float32)
    return bf(np.ldexp(p, n.astype(np.int32)))
ex = lambda x: bf(np.exp(x, dtype=np.float32))
xs = np.linspace(-20, 0, 200001, dtype=np.float32); rel = (exp_srs_np(xs) - ex(xs)) / np.maximum(ex(xs), 1e-30)
i = np.argmax(np.abs(rel)); print(f"exp_srs(sym) rel err: mean {rel.mean():+.2e} rms {np.sqrt((rel**2).mean()):.2e} max {np.abs(rel).max():.2e} at x={xs[i]:.4f} (exp {ex(xs[i]):.5f} got {exp_srs_np(xs[i:i+1])[0]:.5f})")
rel0 = (exp_srs_np(xs, sym=False) - ex(xs)) / np.maximum(ex(xs), 1e-30); print(f"exp_srs(floor f) rel err: mean {rel0.mean():+.2e} rms {np.sqrt((rel0**2).mean()):.2e} max {np.abs(rel0).max():.2e}")
big = np.abs(rel) > 8e-3; print(f"  |rel|>8e-3 fraction {big.mean():.4f}; x of those: {xs[big][:6]}")
HH = 4
def run_model(exp_fn, pfn, sumfn, label, qkv="even"):
    got = np.zeros((HH, NK, 72), np.float32)
    for h in range(HH):
        Qb, Kb, Vb = rq(Q[h]), rq(K[h]), rq(Vp[h].T).T[:NK]
        s = bf(bf(Qb @ Kb.T) * scale)
        e = exp_fn(bf(s - s.max(axis=1, keepdims=True)))
        P = pfn(e); r = bf(1.0 / sumfn(e, P))
        got[h] = bf((P @ Vb) * r)
    dd = np.sqrt(((o[:HH] - got)**2).sum() / (got**2).sum()); dr = np.sqrt(((ref[:HH] - got)**2).sum() / (ref[:HH]**2).sum())
    print(f"model {label:46s}: NPU-vs-model {dd:.2e} | model-vs-emu-ref {dr:.2e}")
sum_e = lambda e, P: e.sum(axis=1, keepdims=True, dtype=np.float32)
sum_P = lambda e, P: P.sum(axis=1, keepdims=True, dtype=np.float32)
ident = lambda e: e
run_model(ex, ident, sum_e, "exact exp, P bf16, sum e")
run_model(exp_srs_np, ident, sum_e, "exp_srs, P bf16, sum e")
run_model(exp_srs_np, lambda e: rq(e, "floor"), sum_e, "exp_srs, P floor, sum e")
run_model(exp_srs_np, lambda e: rq(e, "floor"), sum_P, "exp_srs, P floor, sum P (kernel)")
run_model(ex, lambda e: rq(e, "floor"), sum_P, "exact exp, P floor, sum P")
run_model(exp_srs_np, lambda e: rq(e, "conv_even"), sum_P, "exp_srs, P even, sum P")
