"""attsim (npu-research rev 9): the attention op with the K / V multicast lockstep, for sim16.

Mechanism (design8 + mm8.cc, full12r13f; lowered MLIR build/full12r13f.prj):
  - per core row r one K / V stream: shim r -> memtile r (B<r>, l2_depth objects of 4,608 B) -> ONE MM2S channel multicast to
    the 8 columns' L1 (B<r>L1_<c>, 2 slots each). A multicast send waits for a free L1 slot in ALL 8 destinations, so a column
    runs at most one object ahead of the slowest one: per object the row pays the slowest column.
  - per core: 2 passes (pass 0: tiles 0, 1 = 12 row blocks; pass 1: tile 2 = 6) x 4 head groups (head r + 3 g) x 24 chunks
    x (K object, V object); chunk 23 has no keys (calls only), chunk 22 has 2 of 6 key blocks (the driver's slow path).
  - head-group boundary: the column's A ring (Q objects, broadcast to the column's 3 cores) and the single C object (drained
    before the next attn_begin) make every head group start together: a barrier over the 24 cores + H_p cycles (pre + begin
    + end, r8 PROF stamps: pass 0 8.56 us, pass 1 5.61 us).
  - producer: one fill per object at P_SHIM cycles (shim -> memtile), memtile depth D; send X cycles (4,608 B at 7.95 B/cycle).
Unit cost (cycles per row block of a K object): PREP (rows8 of -m + call, always) + L (the fused loop) + T (exact test) +
R (rescale_rb, when the max moves) + E (aexp_push), switched per variant; V phase VRB per row block. Per-(block, head, column,
tile, rb, chunk) flag / move maps: r14/lazymap.npz (CPU bf16 model, NPU layout).

API: Att(npz).counts(variant) -> per-(S, head, col, pass, chunk) counts; Att.block_cycles(counts, prm) -> mean attention
cycles per block (and per-sequence values). No device; numpy only.
"""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

F = 1.81e3
NCH, NK_CH = 24, 23  # chunks per head group (incl. the empty 23rd), chunks with keys


@dataclass
class Prm:
    PREP: float = 68.0   # rows8(-m) + attk call per row block (r13 bench "rows8 prep + call ~68")
    L: float = 341.0     # fused loop per row block in situ (fitted)
    T: float = 66.0      # exact test (bench)
    R: float = 300.0     # rescale_rb (fitted / bench)
    E: float = 399.0     # aexp_push (FORCE=8 twin)
    VRB: float = 222.0   # V phase per row block (r8 stamps 33.8 us / 276 units)
    OK: float = 60.0     # per K-object call overhead (attn_chunk_k + per-tile driver entry)
    OV: float = 40.0     # per V-object call overhead
    X: float = 580.0     # memtile -> L1 multicast send per object (4,608 B at 7.95 B/cycle)
    P_SHIM: float = 1280.0  # shim -> memtile fill per object (4,608 B at ~3.6 B/cycle)
    D: int = 4           # memtile objects (l2_depth)
    SLOTS: int = 2       # L1 K / V slots per core
    H0: float = 15500.0  # head-group fixed cycles, pass 0 (pre + begin + end)
    H1: float = 10200.0  # pass 1
    LOCK: bool = True    # the multicast lockstep (False = every column independent: the "no lockstep" counterfactual)


class Att:
    def __init__(self, npz, policy=None, frames=None):
        z = np.load(npz)
        if policy is None:
            fl, mv = z["flag"], z["move"]
        else:
            names = list(z["pnames"])
            fl, mv = z["pflag"][names.index(policy)], z["pmove"][names.index(policy)]
        fl, mv = fl.astype(bool), mv.astype(bool)
        if frames is not None:
            fl, mv = fl[frames], mv[frames]
        nfr, nb = fl.shape[:2]
        self.S = nfr * nb
        self.fl = fl.reshape(self.S, 12, 8, 3, 6, NK_CH)  # [S, head, col, tile, rb, chunk]
        self.mv = mv.reshape(self.S, 12, 8, 3, 6, NK_CH)

    def thinned(self, cap=0.109, seed=0):
        """robustness variant: drop random flags (chunks 1-21; flag and move together) on the cores whose slow-unit rate is
        above `cap` (the in-situ count's max 10.9 %) until each is at the cap: the map's column imbalance compressed to the
        measured range."""
        rng = np.random.default_rng(seed)
        fl, mv = self.fl.copy(), self.mv.copy()
        j22 = np.zeros(NK_CH, bool)
        j22[22] = True
        for r in range(3):
            for c in range(8):
                f = fl[:, r::3, c]  # view: [S, 4 heads, tile, rb, chunk]
                m = mv[:, r::3, c]
                rate = (f | j22).mean()
                if rate <= cap:
                    continue
                idx = np.argwhere(f[..., 1:22])
                ndrop = int(round((rate - cap) * f.size))
                pick = idx[rng.choice(len(idx), min(ndrop, len(idx)), replace=False)]
                for q in pick:
                    f[tuple(q[:-1]) + (q[-1] + 1,)] = False
                    m[tuple(q[:-1]) + (q[-1] + 1,)] = False
                fl[:, r::3, c], mv[:, r::3, c] = f, m
        out = Att.__new__(Att)
        out.S, out.fl, out.mv = self.S, fl, mv
        return out

    def remap_rows(self, assign):
        """assign: [8 cols, 3 tiles, 6 rb] -> source (col, tile, rb) index triple array of shape [8, 3, 6, 3]: the row-block
        layout a variant deals to the columns (bit-identical: rows are independent in attention)."""
        a = np.asarray(assign)
        f2 = self.fl[:, :, a[..., 0], a[..., 1], a[..., 2], :]
        m2 = self.mv[:, :, a[..., 0], a[..., 1], a[..., 2], :]
        out = Att.__new__(Att)
        out.S, out.fl, out.mv = self.S, f2, m2
        return out

    def counts(self, v):
        """per-(S, head, col, pass, chunk) row-block counts (n_all, n_L, n_T, n_R, n_E) for a variant dict:
        a (loop on), test: 'none' | 'all' | 'flag', exp: 'none' | 'move' | 'flag' | 'all', j22: chunk 22 slow (default True),
        c0: chunk 0 handling 'flag' (as the map says) | 'qk' (QK-only loop + slow path: loop cost L_QK instead of L)."""
        out = {}
        j22 = np.zeros(NK_CH, bool)
        if v.get("j22", True):
            j22[22] = True
        for p, tiles in ((0, slice(0, 2)), (1, slice(2, 3))):
            fl = self.fl[:, :, :, tiles] | j22
            mv = self.mv[:, :, :, tiles]
            nrb = fl.shape[3] * 6
            shape = fl.shape[:3] + (NK_CH,)
            n_all = np.full(shape, nrb, float)
            n_l = n_all * (1.0 if v.get("a", 1) else 0.0)
            t = v.get("test", "flag")
            n_t = {"none": 0 * n_all, "all": n_all, "flag": fl.sum((3, 4)).astype(float)}[t]
            n_r = 0 * n_all if t == "none" else mv.sum((3, 4)).astype(float)
            e = v.get("exp", "flag")
            n_e = {"none": 0 * n_all, "all": n_all, "flag": fl.sum((3, 4)).astype(float),
                   "move": (mv | j22).sum((3, 4)).astype(float)}[e]
            out[p] = (n_all, n_l, n_t, n_r, n_e)
        return out

    @staticmethod
    def durations(cnt, prm, v=None):
        """per pass: K and V object cycles [S, head, col, 24]"""
        v = v or {}
        res = {}
        for p, (n_all, n_l, n_t, n_r, n_e) in cnt.items():
            lcost = prm.L * n_l
            for j, lj in v.get("chunk_L", {}).items():  # chunk j with another loop cost per row block (QK-only chunk 0,
                lcost = lcost.copy()                    # chunk 22 with nkb trips + zero panels)
                lcost[..., j] = lj * n_l[..., j]
            if "l_scale" in v:
                lcost = lcost * v["l_scale"]
            dk = prm.OK + prm.PREP * n_all * v.get("prep_scale", 1.0) + lcost + v.get("t_scale", 1.0) * prm.T * n_t \
                + v.get("r_scale", 1.0) * prm.R * n_r + v.get("e_scale", 1.0) * prm.E * n_e
            dv = prm.OV + prm.VRB * n_all * v.get("v_scale", 1.0)
            pad = np.full(dk.shape[:-1] + (1,), prm.OK)
            dk = np.concatenate([dk, pad], -1)
            dv = np.concatenate([dv, np.full(dv.shape[:-1] + (1,), prm.OV)], -1)
            res[p] = (dk, dv)
        return res

    def deferred(self, prm, v=None, chunks=range(1, 22)):
        """the 'defer' variant: flagged units of the given chunks leave the lockstep. Returns (counts without their slow work,
        per-pass tail cycles [S, head, col]: their T + R + E, done after the head group's last object, outside the multicast)"""
        v = v or {}
        base = self.counts({**v, "test": "flag", "exp": "flag"})
        tails = {}
        mask = np.zeros(NK_CH, bool)
        mask[list(chunks)] = True
        for p, tiles in ((0, slice(0, 2)), (1, slice(2, 3))):
            fl = self.fl[:, :, :, tiles] & mask
            mv = self.mv[:, :, :, tiles] & mask
            n_all, n_l, n_t, n_r, n_e = base[p]
            nf = fl.sum((3, 4)).astype(float)
            nm = mv.sum((3, 4)).astype(float)
            base[p] = (n_all, n_l, n_t - nf, n_r - nm, n_e - nf)
            tails[p] = (v.get("t_scale", 1.0) * prm.T * nf + v.get("r_scale", 1.0) * prm.R * nm
                        + v.get("e_scale", 1.0) * prm.E * nf).sum(-1)
        return base, tails

    def block_cycles(self, cnt, prm, v=None, detail=False, tail=None):
        """event simulation of one attention op per sequence; returns mean cycles per block. tail[p]: [S, head, col] cycles a
        core adds after its head group's last object (deferred work), before the head-group barrier."""
        dur = self.durations(cnt, prm, v)
        S = self.S
        D, X, PS, NS = prm.D, prm.X, prm.P_SHIM, prm.SLOTS
        nobj = 8 * 2 * NCH
        se = np.zeros((S, 3, nobj))           # send end per object
        done = np.zeros((S, 3, 8, nobj))      # core done per object
        fe = np.zeros((S, 3))                 # producer: last fill end
        t_end = np.zeros(S)
        wait = np.zeros(S)                    # cycles cores spend waiting for objects (sum over cores / 24)
        o = 0
        last = np.zeros((S, 3, 8))
        for p in (0, 1):
            dk, dv = dur[p]
            H = prm.H0 if p == 0 else prm.H1
            for g in range(4):
                B = last.max((1, 2)) + H  # barrier over the 24 cores
                heads = [r + 3 * g for r in range(3)]
                d = np.empty((S, 3, 8, 2 * NCH))
                d[..., 0::2] = dk[:, heads]
                d[..., 1::2] = dv[:, heads]
                prev = np.broadcast_to(B[:, None, None], (S, 3, 8)).copy()
                for k in range(2 * NCH):
                    fs = fe if o < D else np.maximum(fe, se[:, :, o - D])
                    fe = fs + PS
                    slot = done[:, :, :, o - NS] if o >= NS else np.zeros((S, 3, 8))
                    if prm.LOCK:
                        ss = np.maximum.reduce([fe, se[:, :, o - 1] if o else fe * 0, slot.max(2)])
                        arr = (ss + X)[:, :, None]
                        se[:, :, o] = ss + X
                    else:  # each column served on its own (the counterfactual: what the lockstep costs)
                        ss = np.maximum(fe[:, :, None], slot)
                        arr = ss + X
                        se[:, :, o] = arr.max(2)  # memtile slot freed when every column has it
                    st = np.maximum(prev, arr)
                    wait += (st - prev).sum((1, 2)) / 24
                    prev = st + d[..., k]
                    done[:, :, :, o] = prev
                    o += 1
                last = prev if tail is None else prev + tail[p][:, heads]
        t_end = last.max((1, 2)) + 2000.0  # the op's tail (last drain), constant
        if detail:
            return t_end, wait
        return float(t_end.mean())


def default_att():
    return Att(Path(__file__).resolve().parents[1] / "r14" / "lazymap.npz")


V_TWINS = {  # round 13 timing twins (RESULTS 22.2 / 22.1), dispatch ms; D = l2_depth
    "F3": (dict(a=0, test="none", exp="none"), 4, 43.39),
    "F2": (dict(a=1, test="none", exp="none"), 4, 47.14),
    "F16": (dict(a=1, test="all", exp="none"), 4, 48.49),
    "r13c": (dict(a=1, test="all", exp="move"), 4, 49.84),
    "F8": (dict(a=1, test="all", exp="all"), 4, 52.87),
    "r13e": (dict(a=1, test="all", exp="move"), 8, 49.54),
    "r13f": (dict(a=1, test="flag", exp="flag"), 8, 49.21),
}


def step_ms(att, prm, cnts, base):
    """dispatch ms per twin = base + 12 blocks x attention ms per block"""
    out = {}
    for k, (v, d, _) in V_TWINS.items():
        out[k] = base + 12 * att.block_cycles(cnts[k], replace(prm, D=d)) / F / 1e3
    return out
