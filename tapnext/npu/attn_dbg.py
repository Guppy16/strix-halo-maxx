"""Offline hypothesis test for the attention output: which transformation of the reference matches the device O?"""
import numpy as np
d = np.load("/work/npu-tapnext/build/t4_dbg.npz")
got, q, k, vT, o_ref = d["got"], d["q"], d["k"], d["vT"], d["o_ref"]
M_REAL = 1072
rows = np.arange(M_REAL)
def corr(a, b): return np.corrcoef(a.ravel(), b.ravel())[0, 1]
def attn(qh, kh, vh, nkeys=M_REAL, scale=0.125, norm=True):
    s = qh @ kh[:nkeys].T * scale
    s = s - s.max(axis=1, keepdims=True)
    p = np.exp(s)
    o = p @ vh[:nkeys]
    return o / p.sum(axis=1, keepdims=True) if norm else o
g = got[:M_REAL]
print("baseline corr", corr(g, o_ref[:M_REAL]))
# per-row scale: ratio of norms
rat = np.linalg.norm(g, axis=1) / np.maximum(np.linalg.norm(o_ref[:M_REAL], axis=1), 1e-9)
print("row norm ratio got/ref: median %.3f p10 %.3f p90 %.3f" % (np.median(rat), np.percentile(rat, 10), np.percentile(rat, 90)))
hyps = {}
for h in range(12):
    sl = slice(h * 64, h * 64 + 64)
    qh, kh, vh = q[:, sl], k[:, sl], vT[:, sl]
    cands = {
        "ref": attn(qh, kh, vh),
        "unnormalised": attn(qh, kh, vh, norm=False),
        "keys<=1056 (22 chunks)": attn(qh, kh, vh, nkeys=1056),
        "keys<=1104": attn(qh, kh, vh, nkeys=1104),
        "all 1152 keys": attn(qh, kh, vh, nkeys=1152),
        "no scale": attn(qh, kh, vh, scale=1.0),
        "scale 1/64": attn(qh, kh, vh, scale=1 / 64),
        "V blocks transposed": attn(qh, kh, vh.reshape(144, 8, 8, 8).transpose(0, 3, 2, 1).reshape(-1, 64)),
        "K blocks transposed": attn(qh, kh.reshape(144, 8, 8, 8).transpose(0, 3, 2, 1).reshape(-1, 64), vh),
        "sum over keys<1056, O all": (lambda s: (np.exp(s) @ vh[:M_REAL]) / np.exp(s)[:, :1056].sum(1, keepdims=True))((qh @ kh[:M_REAL].T * 0.125) - (qh @ kh[:M_REAL].T * 0.125).max(1, keepdims=True)),
        "sum over keys<1024": (lambda s: (np.exp(s) @ vh[:M_REAL]) / np.exp(s)[:, :1024].sum(1, keepdims=True))((qh @ kh[:M_REAL].T * 0.125) - (qh @ kh[:M_REAL].T * 0.125).max(1, keepdims=True)),
        "sum x 0.714": attn(qh, kh, vh) * 1.4,
    }
    for name, o in cands.items():
        if o is None:
            continue
        hyps.setdefault(name, []).append(corr(g[:, sl], o[:M_REAL]))
for name, cs in hyps.items():
    print(f"{name:32s} mean corr {np.mean(cs):.3f}  per head {[round(c, 2) for c in cs]}")
# per query row: is the device output a scaled version of the reference (corr per row)?
rc = [corr(g[i], o_ref[i]) for i in range(0, M_REAL, 97)]
print("per-row corr samples:", [round(c, 2) for c in rc])
# is it the last-chunk problem: compare against attention over keys of each column separately

print("--- scale curve (corr of got with softmax(QK^T * s) V), s = 0 means uniform attention")
for sc in [0.0, 1 / 256, 1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]:
    cs = []
    for h in range(12):
        sl = slice(h * 64, h * 64 + 64)
        cs.append(corr(g[:, sl], attn(q[:, sl], k[:, sl], vT[:, sl], scale=sc)[:M_REAL]))
    print(f"scale {sc:.5f}: mean corr {np.mean(cs):.3f}")
print("--- dims subsets: scores from the first 32 / last 32 dims only")
for lo, hi in [(0, 32), (32, 64), (0, 8), (0, 16)]:
    cs = []
    for h in range(12):
        sl = slice(h * 64, h * 64 + 64)
        qh, kh = q[:, sl].copy(), k[:, sl].copy()
        qh[:, :lo] = 0; qh[:, hi:] = 0
        cs.append(corr(g[:, sl], attn(qh, kh, vT[:, sl])[:M_REAL]))
    print(f"dims [{lo},{hi}): mean corr {np.mean(cs):.3f}")
print("--- row 0 of head 0: got vs ref (first 8 dims)")
print("got", np.round(g[0, :8], 3).tolist())
print("ref", np.round(o_ref[0, :8], 3).tolist())
print("uniform", np.round(vT[:M_REAL, :8].mean(0), 3).tolist())

print("--- row mapping: for got rows of column 0, the best-matching ref row (same column) and its corr")
c0 = slice(0, 144)
G, R = g[c0], o_ref[c0]
Gn = (G - G.mean(1, keepdims=True)) / G.std(1, keepdims=True)
Rn = (R - R.mean(1, keepdims=True)) / R.std(1, keepdims=True)
C = Gn @ Rn.T / G.shape[1]
best = C.argmax(1)
print("best ref row for got rows 0..23:", best[:24].tolist())
print("corr at best:", np.round(C.max(1)[:24], 2).tolist())
print("corr at identity (row i vs i):", np.round(np.diag(C)[:24], 2).tolist())
print("--- same per head 0 only (64 dims)")
G, R = g[c0, :64], o_ref[c0, :64]
Gn = (G - G.mean(1, keepdims=True)) / G.std(1, keepdims=True)
Rn = (R - R.mean(1, keepdims=True)) / R.std(1, keepdims=True)
C = Gn @ Rn.T / 64
print("head0 best ref row for got rows 0..23:", C.argmax(1)[:24].tolist(), "corr", np.round(C.max(1)[:24], 2).tolist())
# query rows of the SAME tile but the keys... try: got row i vs ref computed with V and K of a different column c'
print("--- keys from a single column only (chunks of column c'): corr of got with attn restricted to keys of column c'")
for cp in [0, 1, 7]:
    cs = []
    for h in range(12):
        sl = slice(h * 64, h * 64 + 64)
        kk = k[:, sl].copy(); kk[np.arange(1152) // 144 != cp] = -1e4  # mask other columns' keys
        cs.append(corr(g[:, sl], attn(q[:, sl], kk, vT[:, sl], nkeys=1152)[:M_REAL]))
    print(f"keys of column {cp} only: mean corr {np.mean(cs):.3f}")
