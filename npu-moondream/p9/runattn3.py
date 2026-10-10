"""Run an attn.py build: 16 heads x 729 queries/keys x 72, bf16 Q/K/V; reference = bf16emu.py attn numerics.
usage: runattn.py BUILD [iters]"""
import json, sys, time
import numpy as np
import pyxrt
from ml_dtypes import bfloat16

b = sys.argv[1]; iters = int(sys.argv[2]) if len(sys.argv) > 2 else 5
cfg = json.load(open(b + ".json"))
QT, HD, KC, NK, ITEMS, reps = cfg["QT"], cfg["HD"], cfg["KC"], cfg["NKEYS"], cfg["ITEMS"], cfg["reps"]
NCH = (NK + KC - 1) // KC; NQT = 768 // QT; H = 16
COLS, ROWS = 8, 4
bf = lambda v: np.asarray(v, np.float32).astype(bfloat16).astype(np.float32)
rng = np.random.default_rng(0)
Q = bf(rng.standard_normal((H, NK, HD)) * 0.5); K = bf(rng.standard_normal((H, NK, HD)) * 0.5); V = bf(rng.standard_normal((H, NK, HD)))
scale = bf(np.float32(1 / HD ** 0.5))
# reference (bf16emu attn): s = bf(bf(QK^T)*scale); p = bf(exp(bf(s - max))); p = bf(p * bf(1/sum)); o = bf(p V)
ref = np.zeros((H, NK, HD), np.float32)
for h in range(H):
    s = bf(bf(Q[h] @ K[h].T) * scale)
    p = bf(np.exp(bf(s - s.max(axis=1, keepdims=True)), dtype=np.float32))
    p = bf(p * bf(1.0 / p.sum(axis=1, keepdims=True, dtype=np.float32)))
    ref[h] = bf(p @ V[h])


def blocked(m):  # (rows, cols) with rows, cols multiples of 8 -> [rb][cb][8][8] flattened
    r, c = m.shape
    return m.reshape(r // 8, 8, c // 8, 8).transpose(0, 2, 1, 3).reshape(-1)


def blocked_T(m):  # each 8x8 block transposed in place: [rb][cb][8 cols][8 rows]
    r, c = m.shape
    return m.reshape(r // 8, 8, c // 8, 8).transpose(0, 2, 3, 1).reshape(-1)


Kp = np.zeros((H, NCH * KC, HD), np.float32); Kp[:, :NK] = K
Vp = np.zeros((H, NCH * KC, HD), np.float32); Vp[:, :NK] = V
Qp = np.zeros((H, NQT * QT, HD), np.float32); Qp[:, :NK] = Q
obj = KC * HD
per_kq = 1 + 2 * NCH
KQ = np.zeros((COLS, ITEMS * per_kq, ROWS, obj), np.float32)
for r in range(ROWS):
    for c in range(COLS):
        core = r * COLS + c
        kq_stream = []
        for i in range(ITEMS):
            item = core * ITEMS + i; h, qt = item // NQT, item % NQT
            qtile = blocked(Qp[h, qt * QT:(qt + 1) * QT])  # [8 rb][9 nb][64]
            kq_stream += [qtile]
            kch = [blocked_T(Kp[h, j * KC:(j + 1) * KC]) for j in range(NCH)]  # K^T blocks [cb][j][8 dims][8 keys]
            vch = [blocked(Vp[h, j * KC:(j + 1) * KC]) for j in range(NCH)]
            for j in range(NCH):
                kq_stream += [kch[j], vch[j]]
        KQ[c, :, r, :] = np.stack(kq_stream)
KQb = np.tile(KQ.reshape(-1).astype(bfloat16), reps)
n_o = COLS * ROWS * ITEMS * obj

dev = pyxrt.device(0); xb = pyxrt.xclbin(b + ".xclbin"); dev.register_xclbin(xb)
ctx = pyxrt.hw_context(dev, xb.get_uuid()); kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
instr = np.fromfile(b + ".bin", dtype=np.uint32)
bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
bo_kq = pyxrt.bo(dev, KQb.nbytes, pyxrt.bo.host_only, kern.group_id(3))
bo_o = pyxrt.bo(dev, n_o * 2, pyxrt.bo.host_only, kern.group_id(4))
bo_i.write(instr.tobytes(), 0); bo_kq.write(KQb.tobytes(), 0)
for q in (bo_i, bo_kq, bo_o): q.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
ts = []
for it in range(iters + 2):
    t0 = time.perf_counter(); run = kern(3, bo_i, len(instr), bo_kq, bo_o)
    st = run.wait(3000000) if it == 0 else run.wait(); ts.append(time.perf_counter() - t0)
    assert str(st) == "ert_cmd_state.ERT_CMD_STATE_COMPLETED", st
bo_o.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
O = np.frombuffer(bo_o.read(n_o * 2, 0), dtype=bfloat16).astype(np.float32).reshape(COLS, ITEMS, ROWS, obj)
out = np.zeros((H, NQT * QT, HD), np.float32)
for r in range(ROWS):
    for c in range(COLS):
        core = r * COLS + c
        for i in range(ITEMS):
            item = core * ITEMS + i; h, qt = item // NQT, item % NQT
            tile = O[c, i, r].reshape(QT // 8, HD // 8, 8, 8).transpose(0, 2, 1, 3).reshape(QT, HD)
            out[h, qt * QT:(qt + 1) * QT] = tile
if "--stats" in sys.argv:
    # debug build: compare per-row sums / maxima / 1/sum with the reference
    got_sum, got_max, got_r = out[:, :NK, 0], out[:, :NK, 1], out[:, :NK, 2]
    for h in range(H):
        s = bf(bf(Q[h] @ K[h].T) * scale); m = s.max(axis=1)
        e = bf(np.exp(bf(s - m[:, None]), dtype=np.float32)); ssum = e.sum(axis=1, dtype=np.float32)
        if h == 0 and "--prof" in sys.argv:
            c = out[h, 0, 3:8]; tot = out[h, 1, 3]; n = out[h, 1, 4]
            print(f"  PROFILE core(head 0, tile 0) kcycles over {n:.0f} chunks: scores {c[0]:.0f} pass0 (max) {c[1]:.0f} pass1 (exp+sums+PV) {c[2]:.0f} | whole-chunk {tot:.0f} = {1e3 * tot / max(n, 1):.0f} cycles/chunk; per pass-1 chunk {1e3 * c[2] / max(n / 2, 1):.0f}, per pass-0 chunk {1e3 * c[1] / max(n / 2, 1):.0f}, scores per chunk {1e3 * c[0] / max(n, 1):.0f}")
        if h < 2:
            rs = np.abs(got_sum[h] - bf(ssum)) / ssum; rm = np.abs(got_max[h] - m)
            print(f"  head {h}: sum rel err median {np.median(rs):.2e} max {rs.max():.2e} | max abs err median {np.median(rm):.2e} max {rm.max():.2e} | ref sum range {ssum.min():.1f}..{ssum.max():.1f} got {got_sum[h].min():.1f}..{got_sum[h].max():.1f}")
            print(f"    first rows ref sums {ssum[:6]} got {got_sum[h, :6]}; ref max {m[:6]} got {got_max[h, :6]}")
    sys.exit(0)
o = out[:, :NK]; err = np.abs(o - ref); exact = ref.astype(bfloat16).astype(np.float32)
print(f"ATTN {H}x{NK}x{HD} reps {reps}: us median {1e6 * np.median(ts[2:]):.0f} min {1e6 * min(ts[2:]):.0f} | mismatch_vs_bf16(ref) {np.mean(o != exact):.3%}"
      f" max_abs {err.max():.3e} (max |ref| {np.abs(ref).max():.2f}) rms_rel {np.sqrt((err**2).sum() / (ref**2).sum()):.2e} nan {np.isnan(o).sum()}")
# reference in the kernel's own order: e = bf(exp(bf(s-max))); O = fp32 sum e v; o = bf(O * bf(1/sum))
ref2 = np.zeros_like(ref)
for h in range(H):
    s = bf(bf(Q[h] @ K[h].T) * scale)
    e = bf(np.exp(bf(s - s.max(axis=1, keepdims=True)), dtype=np.float32))
    r = bf(1.0 / e.sum(axis=1, keepdims=True, dtype=np.float32))
    ref2[h] = bf((e @ V[h]) * r)
err2 = np.abs(o - ref2)
print(f"  vs kernel-order ref: rms_rel {np.sqrt((err2**2).sum() / (ref2**2).sum()):.2e} max_abs {err2.max():.3e}; (emu ref vs kernel-order ref: rms_rel {np.sqrt(((ref - ref2)**2).sum() / (ref**2).sum()):.2e})")
rel_row = [np.sqrt((err2[:, i::8]**2).sum() / (ref2[:, i::8]**2).sum()) for i in range(8)]
print("  rms_rel by row%8:", " ".join(f"{v:.1e}" for v in rel_row))
rel_tile = [np.sqrt((err2[:, qt*64:(qt+1)*64]**2).sum() / (ref2[:, qt*64:(qt+1)*64]**2).sum() + 1e-30) for qt in range(NQT)]
print("  rms_rel by query tile:", " ".join(f"{v:.1e}" for v in rel_tile))
rel_dim = [np.sqrt((err2[:, :, j*8:(j+1)*8]**2).sum() / (ref2[:, :, j*8:(j+1)*8]**2).sum()) for j in range(9)]
print("  rms_rel by dim block:", " ".join(f"{v:.1e}" for v in rel_dim))
rel_head = [np.sqrt((err2[h]**2).sum() / (ref2[h]**2).sum()) for h in range(H)]
print("  rms_rel by head:", " ".join(f"{v:.1e}" for v in rel_head))
