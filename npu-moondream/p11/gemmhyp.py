import sys, numpy as np
d = np.load(sys.argv[1]); c, ex = d["c"], d["exact"]; i = np.load(sys.argv[1].replace(".npz", "") + "_in.npz"); a, w = i["a"].astype(np.float64), i["w"].astype(np.float64)
K = a.shape[1]; kk = np.arange(K)
def score(name, mask):
    ref = a[:, mask] @ w[:, mask].T
    err = np.sqrt(((c - ref) ** 2).sum() / (ref ** 2).sum()); print(f"{name:40s} rms_rel {err:.3e}")
score("full K", kk >= 0)
score("k%64 < 32 (k-subs 0..3)", kk % 64 < 32)
score("k%64 >= 32 (k-subs 4..7)", kk % 64 >= 32)
score("k%16 < 8 (even k-subs)", (kk // 8) % 2 == 0)
score("k%16 >= 8 (odd k-subs)", (kk // 8) % 2 == 1)
score("k < K/2", kk < K // 2)
score("k >= K/2", kk >= K // 2)
score("even 64-chunks", (kk // 64) % 2 == 0)
score("odd 64-chunks", (kk // 64) % 2 == 1)
score("k%64 < 56 (k-subs 0..6)", kk % 64 < 56)
score("k%64 >= 8 (k-subs 1..7)", kk % 64 >= 8)
print("got vs exact, rows 0..15, cols 0..3:")
print(np.array2string(c[:16, :4], precision=3)); print(np.array2string(ex[:16, :4], precision=3))
