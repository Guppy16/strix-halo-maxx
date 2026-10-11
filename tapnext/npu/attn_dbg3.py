import numpy as np
d = np.load("/work/npu-tapnext/build/t4_dbg.npz")
got, o_ref, vT, q, k = d["got"], d["o_ref"], d["vT"], d["q"], d["k"]
M_REAL = 1072
g, r = got[:M_REAL], o_ref[:M_REAL]
for h in (0, 5):
    G, R = g[:, h * 64:(h + 1) * 64], r[:, h * 64:(h + 1) * 64]
    Gn = (G - G.mean(0)) / G.std(0); Rn = (R - R.mean(0)) / R.std(0)
    C = Gn.T @ Rn / G.shape[0]  # [got dim][ref dim]
    print(f"head {h}: best ref dim per got dim:", C.argmax(1).tolist())
    print(f"          corr at best:", np.round(C.max(1), 2).tolist()[:16], "... diag:", np.round(np.diag(C), 2).tolist()[:16])
# hypothesis: output block rows/cols transposed (O[q][d] <-> O[d'][q']) within 8x8 blocks of [8 queries][8 dims]
Rt = r.reshape(134, 8, 12, 8, 8)  # [rb][row][head*8+cb... careful: cols = 768 = 96 blocks of 8
Rt = r.reshape(134, 8, 96, 8).transpose(0, 3, 2, 1).reshape(M_REAL, 768)  # swap row<->col within blocks
print("block-transposed ref corr:", np.corrcoef(g.ravel(), Rt.ravel())[0, 1])
# hypothesis: V dims permuted: O = P V with V's dims permuted within blocks (V^T panel read as [keys][dims])
NK = 48
cs = []
for h in range(12):
    sl = slice(h * 64, h * 64 + 64)
    vv = vT[:NK, sl].reshape(6, 8, 8, 8).transpose(0, 3, 2, 1).reshape(NK, 64)  # [key block][key][dim block][dim] -> swap key<->dim within blocks
    s = q[:, sl] @ k[:NK, sl].T * 0.125; p = np.exp(s - s.max(1, keepdims=True)); o = (p @ vv) / p.sum(1, keepdims=True)
    cs.append(np.corrcoef(g[:, sl].ravel(), o[:M_REAL].ravel())[0, 1])
print("V key<->dim swapped within 8x8 blocks: corr", np.round(np.mean(cs), 3))
# hypothesis: V^T groups (dim blocks) read at a stride of 448 but written at 576 (or vice versa)? emulate: dim block j of the
# device output taken from dims of block j' where byte offset j*448 falls: j' = (j*448)//576 ... use the 24-chunk data? skip
# hypothesis: the ones/sum block mixed into dim block 7? compare per dim block corr
for j in range(8):
    sl = [h * 64 + j * 8 + i for h in range(12) for i in range(8)]
    print(f"dim block {j}: corr {np.corrcoef(g[:, sl].ravel(), r[:, sl].ravel())[0, 1]:.3f}")
