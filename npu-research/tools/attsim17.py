"""attsim17 (npu-research rev 10): attsim's K / V multicast lockstep with per-row-block costs per chunk class, for the round-14
drivers (mm8.cc -DATTKE -DATTKB [-DATTV]).

Per (S, head, col, pass, chunk) the K object costs   OK + sum over the pass's row blocks of cost(chunk class, flagged, moved),
the V object                                          OV + VRB x row blocks.
Chunk classes (r14 driver, mm8.cc chunk_k_fused / k_rb):
  chunk 0        k_rb per row block, forced exact path, no rescale       C0  (bench 971-1,038 per rb)
  chunks 1-21    one batched attkb call; a flagged row block then runs k_rb(force): first-pass rerun FP + exact test T +
                 (rescale R if the max moved) + exact rerun XR                B per rb (bench warm 261)
  chunk 22       k_rb per row block over 2 key blocks + zero panels; flagged ones the exact path   C22 (bench 358)
  chunk 23       no keys (call overhead only)
The r13-era per-rb form (r13f / r14a: one fused call per row block, slow_rb on flags) is variant 'rb'.

Same event model as tools/attsim.py (shim fill, memtile depth D, one multicast send per object, 2 L1 slots per column, the
head-group barrier). Optional knobs for the levers: deal (row-block layout), lock (False = columns independent), dec (decoupled
columns: per-column K / V streams with `slots` L1 objects each), slow-path costs.
"""
from dataclasses import dataclass, field, replace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import attsim as A  # noqa: E402

F = 1.81e3
NCH, NK_CH = 24, 23


@dataclass
class P17:
    # r14 driver, measured on one core (RESULTS 23.3) unless marked
    B: float = 261.0      # batched attkb per unflagged row block, warm (incl. the -m rows prep and flag scan)
    FP: float = 393.0     # k_rb first-pass rerun (6-trip attkb call)
    T: float = 164.0      # exact test (C++ minsize, KRB stamps)
    R: float = 578.0      # rescale_rb incl. call (KRB stamps 637 - 59 for the -m rows store)
    MR: float = 59.0      # -m rows store when the max moves
    XR: float = 393.0     # exact rerun (6-trip attkb, clamp 71.5)
    C0: float = 1005.0    # chunk 0 per row block (bench 971-1,038)
    C22: float = 358.0    # last chunk per row block (2 trips + zero panels)
    VRB: float = 222.0    # V phase per row block (r14c: C++ chunk_v; r14f: 136)
    OK: float = 60.0      # per K object call overhead
    OV: float = 40.0
    X: float = 580.0      # memtile -> L1 multicast send (4,608 B at 7.95 B/cycle)
    P_SHIM: float = 1600.0
    D: int = 8
    SLOTS: int = 2
    H0: float = 15500.0
    H1: float = 10200.0
    LOCK: bool = True
    INSITU: float = 1.0   # in-situ / bench ratio on the K-phase costs (fitted)
    SLOWX: float = 1.0    # extra factor on the slow path only (fitted; 1 = bench)
    # variant switches
    batched: bool = True  # r14 driver (False: the r13 per-rb driver with the costs below)
    ignore_flags: bool = False  # the F128 twin: batched chunks' flags ignored
    RB_L: float = 444.0   # r14a per-rb fused call, unflagged (bench)
    RB_SLOW: float = 426.0  # r14a flagged extra (bench 870 - 444)
    RB_C0: float = 933.0
    RB_C22: float = 321.0


class Att17(A.Att):
    def kcost(self, p: P17):
        """per pass: K-object cycles [S, head, col, 23] (chunks with keys)"""
        res = {}
        for ps, tiles in ((0, slice(0, 2)), (1, slice(2, 3))):
            fl = self.fl[:, :, :, tiles].astype(float)
            mv = self.mv[:, :, :, tiles].astype(float)
            nrb = fl.shape[3] * 6
            nf = fl.sum((3, 4))
            nm = mv.sum((3, 4))
            if p.batched:
                slow = (p.FP + p.T + p.XR) * nf + (p.R + p.MR) * nm
                if p.ignore_flags:
                    slow = 0 * slow
                k = p.B * nrb + p.SLOWX * slow
                k[..., 0] = p.C0 * nrb
                k[..., 22] = p.C22 * nrb + p.SLOWX * ((p.FP + p.T + p.XR) * nf[..., 22] + (p.R + p.MR) * nm[..., 22])
            else:
                k = p.RB_L * nrb + p.SLOWX * p.RB_SLOW * nf
                k[..., 0] = p.RB_C0 * nrb
                k[..., 22] = p.RB_C22 * nrb + p.SLOWX * p.RB_SLOW * nf[..., 22]
            res[ps] = (p.OK + p.INSITU * k, nrb)
        return res

    def sim(self, p: P17, detail=False, dec=False, tail=None):
        """event simulation; dec=True: each column has its own K / V stream (no multicast coupling), producer shared per
        core row (one shim fill per object per column: P_SHIM / 8 per column-object when the shim serves 8 streams)"""
        kc = self.kcost(p)
        S = self.S
        dur = {}
        for ps, (dk, nrb) in kc.items():
            dk = np.concatenate([dk, np.full(dk.shape[:-1] + (1,), p.OK)], -1)
            dv = np.full(dk.shape, p.OV + p.VRB * nrb)
            dv[..., -1] = p.OV
            dur[ps] = (dk, dv)
        D, X, PS, NS = p.D, p.X, p.P_SHIM, p.SLOTS
        nobj = 8 * 2 * NCH
        se = np.zeros((S, 3, 8, nobj)) if dec else np.zeros((S, 3, nobj))
        done = np.zeros((S, 3, 8, nobj))
        fe = np.zeros((S, 3, 8)) if dec else np.zeros((S, 3))
        o = 0
        last = np.zeros((S, 3, 8))
        wait = np.zeros(S)
        for ps in (0, 1):
            dk, dv = dur[ps]
            H = p.H0 if ps == 0 else p.H1
            for g in range(4):
                Bt = last.max((1, 2)) + H
                heads = [r + 3 * g for r in range(3)]
                d = np.empty((S, 3, 8, 2 * NCH))
                d[..., 0::2] = dk[:, heads]
                d[..., 1::2] = dv[:, heads]
                prev = np.broadcast_to(Bt[:, None, None], (S, 3, 8)).copy()
                for k in range(2 * NCH):
                    slot = done[..., o - NS] if o >= NS else np.zeros((S, 3, 8))
                    if dec:
                        # per column: own memtile queue of depth D, fill at PS (the shim / DDR share is checked separately)
                        fs = fe if o < D else np.maximum(fe, se[..., o - D])
                        fe = fs + PS
                        ss = np.maximum.reduce([fe, se[..., o - 1] if o else fe * 0, slot])
                        arr = ss + X
                        se[..., o] = arr
                    else:
                        fs = fe if o < D else np.maximum(fe, se[:, :, o - D])
                        fe = fs + PS
                        if p.LOCK:
                            ss = np.maximum.reduce([fe, se[:, :, o - 1] if o else fe * 0, slot.max(2)])
                            arr = (ss + X)[:, :, None]
                            se[:, :, o] = ss + X
                        else:
                            ss = np.maximum(fe[:, :, None], slot)
                            arr = ss + X
                            se[:, :, o] = arr.max(2)
                    st = np.maximum(prev, arr)
                    wait += (st - prev).sum((1, 2)) / 24
                    prev = st + d[..., k]
                    done[..., o] = prev
                    o += 1
                last = prev if tail is None else prev + tail[ps][:, heads]
        t_end = last.max((1, 2)) + 2000.0
        if detail:
            return t_end, wait
        return float(t_end.mean())

    def own(self, p: P17):
        """mean per-core own K + V cycles per block (no waiting): the 'columns independent, no barrier imbalance' floor"""
        kc = self.kcost(p)
        tot = 0
        for ps, (dk, nrb) in kc.items():
            tot = tot + dk.sum(-1).sum(1) + 23 * (p.OV + p.VRB * nrb) * 12  # [S, col]
        return tot


def load(thin=None, policy=None):
    a = A.Att(Path(__file__).resolve().parents[1] / "r14" / "lazymap.npz", policy=policy)
    if thin is not None:
        a = a.thinned(thin)
    out = Att17.__new__(Att17)
    out.S, out.fl, out.mv = a.S, a.fl, a.mv
    return out


def remap(att, assign):
    b = att.remap_rows(assign)
    out = Att17.__new__(Att17)
    out.S, out.fl, out.mv = b.S, b.fl, b.mv
    return out


def ms(att, p, **kw):
    return 12 * att.sim(p, **kw) / F / 1e3
