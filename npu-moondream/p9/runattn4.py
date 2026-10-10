"""Run an attn.py build: 16 heads x 729 queries/keys x 72, bf16 Q/K/V; reference = bf16emu.py attn numerics.
usage: runattn.py BUILD [iters]"""
import json, sys, time
import numpy as np
import pyxrt
from aie.utils import bfp
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


def to_panels(flat_blocks):  # blocked bf16 values (n_blocks x 64, rows of 8) -> bfp16ebs8 panels as bytes (9 B per row of 8)
    b = np.asarray(flat_blocks, np.float32).reshape(-1, 8)  # every 8 consecutive values = one panel row = one exponent block
    return np.ascontiguousarray(bfp.encode(b, rounding="conv_even")).reshape(-1)


def groups(parts, gbytes, total):  # concatenate panel groups, each padded to gbytes (64 B-aligned seeks), then to total bytes
    out = np.zeros(total, np.uint8); off = 0
    for p in parts:
        assert len(p) <= gbytes and off + gbytes <= total
        out[off:off + len(p)] = p; off += gbytes
    return out.view(np.uint32)


def qk_obj(t):  # 48 x 72 (rows x dims) -> [6 row blocks][9 dim panels], 704 B per group
    return groups([to_panels(blocked(t[rb * 8:(rb + 1) * 8])) for rb in range(t.shape[0] // 8)], 704, 4224)


def v_obj(v):  # 48 keys x 72 dims -> [9 dim blocks][6 key blocks] V^T panels [8 dims][8 keys], 448 B per group
    return groups([to_panels(np.concatenate([v[cb * 8:(cb + 1) * 8, j * 8:(j + 1) * 8].T.reshape(-1) for cb in range(v.shape[0] // 8)]))
                   for j in range(v.shape[1] // 8)], 448, 4224)


def dec_panels(words):
    return bfp.decode(np.ascontiguousarray(np.asarray(words)).view(np.uint8).reshape(-1, 9)).reshape(-1)


Kp = np.zeros((H, NCH * KC, HD), np.float32); Kp[:, :NK] = K
Vp = np.zeros((H, NCH * KC, HD), np.float32); Vp[:, :NK] = V
Qp = np.zeros((H, NQT * QT, HD), np.float32); Qp[:, :NK] = Q
obj = 4224 // 4  # words per bfp object (6 padded groups)
oobj = QT * HD
per_kq = 1 + 2 * NCH
KQ = np.zeros((COLS, ITEMS * per_kq, ROWS, obj), np.uint32)
for r in range(ROWS):
    for c in range(COLS):
        core = r * COLS + c
        kq_stream = []
        for i in range(ITEMS):
            item = core * ITEMS + i; h, qt = item // NQT, item % NQT
            qtile = qk_obj(Qp[h, qt * QT:(qt + 1) * QT])
            kq_stream += [qtile]
            kch = [qk_obj(Kp[h, j * KC:(j + 1) * KC]) for j in range(NCH)]  # K natural [keys][dims] (op_transpose in the mmul)
            vch = [v_obj(Vp[h, j * KC:(j + 1) * KC]) for j in range(NCH)]
            for j in range(NCH):
                kq_stream += [kch[j], vch[j]]
        KQ[c, :, r, :] = np.stack(kq_stream)
KQb = np.tile(KQ.reshape(-1), reps)
n_o = COLS * ROWS * ITEMS * oobj

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
O = np.frombuffer(bo_o.read(n_o * 2, 0), dtype=bfloat16).astype(np.float32).reshape(COLS, ITEMS, ROWS, oobj)
RB, CB = QT // 8, KC // 8
ex = lambda d: bf(np.exp(d, dtype=np.float32))
def exp_srs_np(x):  # the kernel's exp: z = x log2e via bf16 hi/lo MACs, q = round(256 z), 2^f by a bf16 Horner cubic
    xh = bf(x); xl = bf(x - xh)
    L = np.float32(1.4426950408889634); LH = bf(np.array([L], np.float32))[0]; LL = np.float32(L - LH)
    za = (xh * LH + xh * LL + xl * LH).astype(np.float32)
    qz = np.clip(np.rint(za * 256), -60 * 256, 60 * 256)
    n = np.floor((qz + 128) / 256); fr = qz - n * 256; f = (fr / 256).astype(np.float32)  # symmetric fraction (v4b)
    p = bf(f * bf(np.array([0.05550411], np.float32))[0])
    p = bf(np.float32(0.24022651) + f * p); p = bf(np.float32(0.69314718) + f * p); p = (np.float32(1.0) + f * p).astype(np.float32)
    return bf(np.ldexp(p, n.astype(np.int32)))
def rq_floor(m):
    f = np.asarray(m, np.float32); f = f.reshape(-1, 8) if f.shape[-1] % 8 == 0 else np.concatenate([f, np.zeros((f.shape[0], (-f.shape[1]) % 8), np.float32)], axis=1).reshape(-1, 8)
    return bfp.decode(bfp.encode(f, rounding="floor")).reshape(m.shape[0], -1)[:, :m.shape[1]]

if "--dump" in sys.argv:  # DEBUG_DUMP build: chunk-0 scores (rb 0..3) and P panels (all rb) vs the bfp-operand reference
    raw = np.frombuffer(bo_o.read(n_o * 2, 0), dtype=np.uint16).reshape(COLS, ITEMS, ROWS, oobj)
    rq = lambda m: dec_panels(to_panels(np.asarray(m, np.float32).reshape(-1, 8))).reshape(m.shape)
    for (c, i, r_) in [(0, 0, 0), (1, 0, 0), (0, 1, 2)]:
        core = r_ * COLS + c; item = core * ITEMS + i; h, qt = item // NQT, item % NQT
        Qb, Kb = rq(Qp[h, qt * QT:(qt + 1) * QT]), rq(Kp[h, :KC])
        s = bf(bf(Qb @ Kb.T) * scale)                       # [48 q][48 keys]
        e = bf(np.exp(bf(s - s.max(axis=1, keepdims=True)), dtype=np.float32))
        got_s = O[c, i, r_][:4 * CB * 64].reshape(4, CB, 8, 8).transpose(0, 2, 1, 3).reshape(32, KC)
        got_p = dec_panels(raw[c, i, r_][4 * CB * 64:4 * CB * 64 + RB * CB * 72 // 2].view(np.uint32)).reshape(RB, CB, 8, 8).transpose(0, 2, 1, 3).reshape(QT, KC)
        ds = np.abs(got_s - s[:32]); dp = np.abs(got_p - rq(e))
        print(f"  item core {core} i {i} (head {h} qt {qt}): scores max_abs_err {np.nanmax(ds):.3e} nan {np.isnan(got_s).sum()} (|s| max {np.abs(s).max():.2f}) | P max_abs_err {np.nanmax(dp):.3e} nan {np.isnan(got_p).sum()}")
        okmap = (ds.reshape(4, 8, CB, 8) < 0.02).mean(axis=(0, 2)); pmap = (dp.reshape(RB, 8, CB, 8) < 0.03).mean(axis=(0, 2))
        d32 = bf(got_s - got_s.max(axis=1, keepdims=True))  # P model from the kernel's own scores: exp_srs then bfp
        for lab, pm in (("bfp(exp_srs) conv_even", rq(exp_srs_np(d32))), ("bfp(exp_srs) floor", rq_floor(exp_srs_np(d32))), ("bfp(exact exp) conv_even", rq(ex(d32)))):
            dd = got_p[:32] - pm; print(f"    P vs {lab:26s}: max_abs {np.abs(dd).max():.2e} rms {np.sqrt((dd**2).mean()):.2e} mean {dd.mean():+.2e} exact-frac {(dd == 0).mean():.3f}")
        if (c, i, r_) == (0, 0, 0):
            rawp = raw[c, i, r_][4 * CB * 64:4 * CB * 64 + RB * CB * 72 // 2].view(np.uint8).reshape(RB, CB, 8, 9)  # [rb][cb][row][9 B]
            em = exp_srs_np(d32)  # model e before bfp, rows 0..31
            for (rb, cb, row) in ((0, 0, 0), (0, 1, 3), (1, 2, 5), (3, 5, 7)):
                vals = em[rb * 8 + row, cb * 8:(cb + 1) * 8]
                he = bfp.encode(vals.reshape(1, 8), rounding="conv_even").reshape(-1); hf = bfp.encode(vals.reshape(1, 8), rounding="floor").reshape(-1)
                print(f"    group rb{rb} cb{cb} row{row}: e(model) {np.array2string(vals, precision=4)}\n      kernel bytes {rawp[rb, cb, row].tolist()} -> {np.array2string(dec_panels(rawp[rb, cb, row].copy()), precision=4)}\n      host even    {he.tolist()}\n      host floor   {hf.tolist()}")
        print("    scores ok-fraction per (row-in-block, key-in-block):\n" + np.array2string(okmap, precision=2, max_line_width=200))
        print("    scores ok-fraction per (rb, cb):\n" + np.array2string((ds.reshape(4, 8, CB, 8) < 0.02).mean(axis=(1, 3)), precision=2, max_line_width=200))
        print("    P ok-fraction per (row-in-block, key-in-block):\n" + np.array2string(pmap, precision=2, max_line_width=200))
        print("    block(0,0) err:\n" + np.array2string(ds[:8, :8], precision=2, max_line_width=200))
        print("    scores max err per row (32):", np.array2string(np.nanmax(ds, axis=1), precision=2, max_line_width=200))
        print("    scores max err per key (48):", np.array2string(np.nanmax(ds, axis=0), precision=2, max_line_width=250))
        print("    P max err per row (48):", np.array2string(np.nanmax(dp, axis=1), precision=2, max_line_width=250))
        print("    s ref row0", np.array2string(s[0, :8], precision=3), "got", np.array2string(got_s[0, :8], precision=3))
        print("    s ref row9", np.array2string(s[9, :8], precision=3), "got", np.array2string(got_s[9, :8], precision=3))
        print("    P ref row0", np.array2string(rq(e)[0, :8], precision=3), "got", np.array2string(got_p[0, :8], precision=3))
        print("    P ref row40", np.array2string(rq(e)[40, 8:16], precision=3), "got", np.array2string(got_p[40, 8:16], precision=3))
    sys.exit(0)
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
o = out[:, :NK]
if np.isnan(o).any():
    nanmap = np.isnan(out)
    print("  NaN fraction by head:", " ".join(f"{v:.2f}" for v in nanmap.mean(axis=(1, 2))))
    print("  NaN fraction by query tile:", " ".join(f"{nanmap[:, qt*QT:(qt+1)*QT].mean():.2f}" for qt in range(NQT)))
    print("  NaN fraction by row%8:", " ".join(f"{nanmap[:, i::8].mean():.2f}" for i in range(8)))
    print("  NaN fraction by dim block:", " ".join(f"{nanmap[:, :, j*8:(j+1)*8].mean():.2f}" for j in range(9)))
    fin = np.isfinite(o); errf = np.abs(np.where(fin, o, 0) - np.where(fin, ref, 0))
    print(f"  finite outputs: rms_rel {np.sqrt((errf**2).sum() / (np.where(fin, ref, 0)**2).sum()):.2e} max_abs {errf.max():.3e}")
err = np.abs(o - ref); exact = ref.astype(bfloat16).astype(np.float32)
print(f"ATTN {H}x{NK}x{HD} reps {reps}: us median {1e6 * np.median(ts[2:]):.0f} min {1e6 * min(ts[2:]):.0f} | mismatch_vs_bf16(ref) {np.mean(o != exact):.3%}"
      f" max_abs {err.max():.3e} (max |ref| {np.abs(ref).max():.2f}) rms_rel {np.sqrt((err**2).sum() / (ref**2).sum()):.2e} nan {np.isnan(o).sum()}")
# reference in the kernel's own order: e = bf(exp(bf(s-max))); O = fp32 sum e v; o = bf(O * bf(1/sum))
# reference with bfp16-rounded operands (what the NPU actually multiplies)
def rq(m):  # bfp16 round-trip; rows padded to a multiple of 4 panel rows for the uint32 view
    f = np.asarray(m, np.float32); f = f.reshape(-1, 8) if f.shape[-1] % 8 == 0 else np.concatenate([f, np.zeros((f.shape[0], (-f.shape[1]) % 8), np.float32)], axis=1).reshape(-1, 8); n = f.shape[0]; pad = (-n) % 4
    g = np.concatenate([f, np.zeros((pad, 8), np.float32)]) if pad else f
    d = dec_panels(to_panels(g).view(np.uint32))[:n * 8].reshape(m.shape[0], -1)
    return d[:, :m.shape[1]]
ref3 = np.zeros_like(ref)
for h in range(H):
    Qb, Kb, Vb = rq(Q[h]), rq(K[h]), rq(Vp[h].T).T[:NK]  # V's exponent groups run along keys (V^T panels)
    s = bf(bf(Qb @ Kb.T) * scale)
    e = bf(np.exp(bf(s - s.max(axis=1, keepdims=True)), dtype=np.float32))
    r = bf(1.0 / e.sum(axis=1, keepdims=True, dtype=np.float32))
    ref3[h] = bf((rq(e) @ Vb) * r)
err3 = np.abs(o - ref3)
print(f"  vs bfp-operand ref: rms_rel {np.sqrt((err3**2).sum() / (ref3**2).sum()):.2e} max_abs {err3.max():.3e}")
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
if "--expdump" in sys.argv:  # DEBUG_EXP build: o[0..1536) of item 0 = exp32(-i/64)
    rawo = np.frombuffer(bo_o.read(n_o * 2, 0), dtype=bfloat16).astype(np.float32).reshape(COLS, ITEMS, ROWS, oobj)
    got = rawo[0, 0, 0, :1536]; xs = bf(-(np.arange(1536, dtype=np.float32)) / 64); exact = bf(np.exp(xs, dtype=np.float32))  # the kernel's d is bf16
    rel = (got - exact) / exact
    print(f"  exp32 ramp: rel err mean {rel.mean():+.2e} rms {np.sqrt((rel**2).mean()):.2e} max {np.abs(rel).max():.2e}; worst at x={xs[np.argmax(np.abs(rel))]:.4f}")
    for x0 in (0.0, -0.25, -0.5, -0.75, -1.0, -1.5, -2.5, -10.0):
        i = int(-x0 * 64); print(f"    x={x0:6.3f} exact {exact[i]:.5f} got {got[i]:.5f} rel {rel[i]:+.2e}")
    fr = (xs * 1.4426950408889634); fr = fr - np.rint(fr)
    for lo, hi in ((-0.5, -0.25), (-0.25, 0), (0, 0.25), (0.25, 0.5)):
        sel = (fr >= lo) & (fr < hi); print(f"    f in [{lo},{hi}): mean rel {rel[sel].mean():+.2e} max {np.abs(rel[sel]).max():.2e}")
    for lo, hi in ((-2, 0), (-8, -2), (-16, -8), (-24, -16)):
        sel = (xs > lo) & (xs <= hi); print(f"    x in ({lo},{hi}]: mean rel {rel[sel].mean():+.2e} rms {np.sqrt((rel[sel]**2).mean()):.2e} max {np.abs(rel[sel]).max():.2e}")
    gn = rawo[0, 0, 0, 1536:2048]; gf = rawo[0, 0, 0, 2048:2560]; gzr = rawo[0, 0, 0, 2560:3072]; z = xs[:512] * np.float32(1.4426950408889634)
    print(f"    n vs round(z): exact-frac {(gn == np.rint(z)).mean():.3f}; n - round(z) histogram {np.unique(gn - np.rint(z), return_counts=True)}")
    print(f"    zr vs z: |zr - z| max {np.abs(gzr - z).max():.4f} (1/256 = 0.0039); f vs zr - n: max |f - (zr - n)| {np.abs(gf - (gzr - gn)).max():.4f}; f range {gf.min():.3f}..{gf.max():.3f}")
    for i in (16, 17, 33, 100, 200, 400, 511): print(f"      x={xs[i]:.4f} z={z[i]:.4f} n={gn[i]:.0f} zr={gzr[i]:.4f} f={gf[i]:.4f} e={got[i]:.5f} exact={exact[i]:.5f}")
    ulp = np.abs(got - exact) / (np.abs(exact) * 2.0**-7); print(f"    |err| in bf16 ulps: <=0.5: {(ulp <= 0.5).mean():.3f}, <=1: {(ulp <= 1.01).mean():.3f}, max {ulp.max():.2f}")
    sys.exit(0)
if "--save" in sys.argv:
    np.savez(b + "_out.npz", o=o, ref=ref, Q=Q, K=K, V=V, Vp=Vp, scale=scale, NK=NK)
if "--model" in sys.argv:  # where does the kernel's error come from? host models of each v4 numerics step, heads 0..3
    HH = 4
    def run_model(exp_fn, pround, label):
        got = np.zeros((HH, NK, HD), np.float32)
        for h in range(HH):
            Qb, Kb, Vb = rq(Q[h]), rq(K[h]), rq(Vp[h].T).T[:NK]
            s = bf(bf(Qb @ Kb.T) * scale)
            e = exp_fn(bf(s - s.max(axis=1, keepdims=True)))
            r = bf(1.0 / e.sum(axis=1, keepdims=True, dtype=np.float32))
            got[h] = bf((pround(e) @ Vb) * r)
        d = np.sqrt(((o[:HH] - got)**2).sum() / (got**2).sum()); dr = np.sqrt(((ref[:HH] - got)**2).sum() / (ref[:HH]**2).sum())
        print(f"  model {label:38s}: rms_rel NPU-vs-model {d:.2e} | model-vs-emu-ref {dr:.2e}")
    ex = lambda d: bf(np.exp(d, dtype=np.float32))
    run_model(ex, lambda e: e, "exact exp, P bf16 (no bfp)")
    run_model(ex, rq, "exact exp, P bfp conv_even")
    run_model(ex, rq_floor, "exact exp, P bfp floor")
    run_model(exp_srs_np, lambda e: e, "exp_srs, P bf16")
    run_model(exp_srs_np, rq, "exp_srs, P bfp conv_even")
    run_model(exp_srs_np, rq_floor, "exp_srs, P bfp floor")
    e_ex = ex(np.linspace(-20, 0, 100001, dtype=np.float32)); e_k = exp_srs_np(np.linspace(-20, 0, 100001, dtype=np.float32))
    rel = (e_k - e_ex) / np.maximum(e_ex, 1e-30)
    print(f"  exp_srs vs bf(exp): rel err mean {rel.mean():+.2e} rms {np.sqrt((rel**2).mean()):.2e} max {np.abs(rel).max():.2e}")
