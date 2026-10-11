"""48-key single-chunk attention: which transformation matches the device output?"""
import numpy as np
d = np.load("/work/npu-tapnext/build/t4_dbg.npz")
got, q, k, vT, o_ref = d["got"], d["q"], d["k"], d["vT"], d["o_ref"]
M_REAL, NK = 1072, 48
g = got[:M_REAL]
def corr(a, b): return np.corrcoef(a.ravel(), b.ravel())[0, 1]
def rr(a, b): return np.sqrt(((a - b) ** 2).sum() / (b ** 2).sum())
def bfp_rows(p):  # quantise each row's 8-key blocks to 8-bit signed mantissas at the block max exponent (device P panels)
    out = np.zeros_like(p)
    for b in range(0, p.shape[1], 8):
        blk = p[:, b:b + 8]
        amax = np.abs(blk).max(1, keepdims=True)
        e = np.where(amax > 0, np.floor(np.log2(np.maximum(amax, 1e-38))) - 6, -133.0)
        sc = np.exp2(e)
        out[:, b:b + 8] = np.clip(np.rint(blk / sc), -128, 127) * sc
    return out
def attn(qh, kh, vh, perm=None, quant=False, scale=0.125, maxmode="row"):
    kk = kh[:NK] if perm is None else kh[:NK][perm]
    vv = vh[:NK] if perm is None else vh[:NK][perm]
    s = qh @ kk.T * scale
    s = s.astype(np.float32)
    m = s.max(1, keepdims=True) if maxmode == "row" else s.max()
    p = np.exp(s - m)
    if quant:
        p = bfp_rows(p)
    return (p @ vv) / p.sum(1, keepdims=True)
cands = {
    "ref": dict(),
    "bfp P": dict(quant=True),
    "keys reversed": dict(perm=np.arange(NK)[::-1]),
    "key blocks reversed": dict(perm=np.arange(NK).reshape(6, 8)[::-1].reshape(-1)),
    "keys reversed in blocks": dict(perm=np.arange(NK).reshape(6, 8)[:, ::-1].reshape(-1)),
    "key blocks pairs swapped": dict(perm=np.arange(NK).reshape(3, 2, 8)[:, ::-1].reshape(-1)),
    "no scale": dict(scale=1.0),
    "scale/2": dict(scale=0.0625),
}
res = {}
for name, kw in cands.items():
    cs, rs = [], []
    for h in range(12):
        sl = slice(h * 64, h * 64 + 64)
        o = attn(q[:, sl], k[:, sl], vT[:, sl], **kw)[:M_REAL]
        cs.append(corr(g[:, sl], o)); rs.append(rr(g[:, sl], o))
    print(f"{name:28s} corr {np.mean(cs):.3f} rms_rel {np.mean(rs):.3f}")
# per-row scale / per query tile / per row within the tile
ratio = np.linalg.norm(g, axis=1) / np.linalg.norm(o_ref[:M_REAL], axis=1)
rows = np.arange(M_REAL)
print("row norm ratio by query tile:", [f"{np.median(ratio[rows % 144 // 48 == t]):.3f}" for t in range(3)])
print("row norm ratio by row within 8-block:", [f"{np.median(ratio[rows % 8 == i]):.3f}" for i in range(8)])
print("row norm ratio by rb within tile:", [f"{np.median(ratio[rows % 48 // 8 == i]):.3f}" for i in range(6)])
# are some dims systematically wrong? per dim block rms_rel
e = g - o_ref[:M_REAL]
print("rms_rel by dim block j:", [f"{np.sqrt((e[:, (np.arange(768) % 64) // 8 == j] ** 2).sum() / (o_ref[:M_REAL][:, (np.arange(768) % 64) // 8 == j] ** 2).sum()):.2f}" for j in range(8)])
print("got row0 head0:", np.round(g[0, :8], 3).tolist())
print("ref row0 head0:", np.round(o_ref[0, :8], 3).tolist())
# hypothesis: the device output equals attention with keys of a DIFFERENT query tile's chunk? (e.g. chunk index = tile)
for t in range(3):
    cs = []
    for h in range(12):
        sl = slice(h * 64, h * 64 + 64)
        kk, vv = k[48 * t:48 * t + 48, sl], vT[48 * t:48 * t + 48, sl]
        s = q[:, sl] @ kk.T * 0.125; p = np.exp(s - s.max(1, keepdims=True)); o = (p @ vv) / p.sum(1, keepdims=True)
        cs.append(corr(g[:, sl], o[:M_REAL]))
    print(f"keys = chunk {t} (keys {48*t}..): corr {np.mean(cs):.3f}")
