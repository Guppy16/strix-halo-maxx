"""rev 10: re-fit the lockstep model with the round-14 MEASURED slow-path costs (exact test 164, rescale 578, reruns 393).

The rev-9 fit (attcal.py) had T and R degenerate and landed on T 66 / R 300 with the raw map; the measured T + 0.1 R is 222, not
96, so the raw map's slow units are too many or too clustered for the measured costs. Here the map's chunk 1-21 flags are kept
with probability kappa (uniform random thinning, 4 seeds averaged) and one kappa is fitted on:
  r13 twins (old per-rb driver; attsim.Prm with T 164, R 578 + 59 (the -m store is r14's; r13 had rows8 inside PREP: R 578),
             E fitted, L fitted, P_SHIM fitted) and
  r14 deltas: F128 - r14a -2.56, r14c - F128 +2.08 .. +2.53, r14e - r14cz -0.80 (ATTV only; VTK's -0.11 is in qkv).
    python3 fit17.py
"""
from dataclasses import replace
import itertools

import numpy as np

import attsim as A0
import attsim17 as A

SEEDS = (0, 1, 2, 3)


def kappa_att(base, kappa, seed):
    rng = np.random.default_rng(seed)
    fl, mv = base.fl.copy(), base.mv.copy()
    keep = rng.random(fl.shape) < kappa
    keep[..., 0] = True
    keep[..., 22] = True
    fl &= keep
    mv &= keep
    out = A.Att17.__new__(A.Att17)
    out.S, out.fl, out.mv = base.S, fl, mv
    return out


def r13_pred(att, prm):
    raw = {}
    for k, (v, d, _) in A0.V_TWINS.items():
        raw[k] = 12 * A0.Att.block_cycles(att, att.counts(v), replace(prm, D=d)) / A.F / 1e3
    return raw


def r14_pred(att, p):
    return {
        "r14a": A.ms(att, replace(p, batched=False)),
        "r14c": A.ms(att, p),
        "F128": A.ms(att, replace(p, ignore_flags=True)),
        "r14e_V": A.ms(att, replace(p, VRB=136.0)) - A.ms(att, p),
    }


R13FIT = ["F3", "F2", "F16", "r13c", "F8"]
MEAS13 = {k: m for k, (_, _, m) in A0.V_TWINS.items()}


def evaluate(atts, prm13, p14):
    r13 = [r13_pred(a, prm13) for a in atts]
    r13 = {k: np.mean([r[k] for r in r13]) for k in r13[0]}
    base = np.mean([MEAS13[k] - r13[k] for k in R13FIT])
    e13 = {k: base + r13[k] - MEAS13[k] for k in r13}
    r14 = [r14_pred(a, p14) for a in atts]
    r14 = {k: np.mean([r[k] for r in r14]) for k in r14[0]}
    d = {"F128-r14a": (r14["F128"] - r14["r14a"], -2.56), "r14c-F128": (r14["r14c"] - r14["F128"], 2.30),
         "V": (r14["r14e_V"], -0.80)}
    return e13, d


if __name__ == "__main__":
    raw = A.load()
    print("kappa  | r13 twins err (F3 F2 F16 r13c F8 | r13e r13f) rms | r14: F128-r14a (-2.56)  r14c-F128 (+2.08..2.53)  V (-0.80)")
    best = []
    for kappa in (1.0, 0.8, 0.7, 0.6, 0.5, 0.4):
        atts = [kappa_att(raw, kappa, s) for s in (SEEDS if kappa < 1 else (0,))]
        for L, E, T, b in itertools.product((380, 400, 420), (360, 400), (66, 100, 130, 164), (261,)):
            prm13 = replace(A0.Prm(), L=L, T=T, R=578, E=E, P_SHIM=1600)
            p14 = replace(A.P17(), B=b)
            e13, d = evaluate(atts, prm13, p14)
            rms13 = np.sqrt(np.mean([e13[k] ** 2 for k in R13FIT]))
            r14err = np.sqrt(np.mean([(m - t) ** 2 for m, t in d.values()]))
            best.append((rms13 + r14err, kappa, L, E, T, e13, d, rms13))
        cand = [x for x in best if x[1] == kappa]
        s, kappa_, L, E, b, e13, d, rms13 = min(cand, key=lambda x: x[0])
        print(f"{kappa:4.2f} L{L} E{E} T{b} | " + " ".join(f"{e13[k]:+.2f}" for k in R13FIT) + " | "
              + " ".join(f"{e13[k]:+.2f}" for k in ("r13e", "r13f")) + f" rms {rms13:.3f} | "
              + "  ".join(f"{m:+.2f}" for m, _ in d.values()))
