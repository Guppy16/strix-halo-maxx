"""rev 10: does attsim's lockstep mechanism, fed the round-14 one-core costs, reproduce the round-14 builds and the F128 twin?
    python3 cal17.py
Builds (RESULTS 23.1, dispatch medians): r13f 49.23, r14a 48.71, r14c 48.68 (min 48.23; runs 48.24-48.73), F128 46.15,
r14cz 48.16, r14e 47.25 (r14f 47.31). Deltas only (the base is the rest of the step)."""
from dataclasses import replace

import numpy as np

import attsim17 as A

MEAS = {"r14a": 48.71, "r14c": 48.68, "r14c_lo": 48.24, "F128": 46.15, "r14cz": 48.16, "r14e": 47.25}


def variants(p):
    r14a = replace(p, batched=False)
    r14c = replace(p)
    f128 = replace(p, ignore_flags=True)
    r14e = replace(p, VRB=136.0)
    return dict(r14a=r14a, r14c=r14c, F128=f128, r14e=r14e)


def run(att, p, tag):
    v = variants(p)
    r = {k: A.ms(att, q) for k, q in v.items()}
    nl = {k: A.ms(att, replace(q, LOCK=False)) for k, q in v.items()}
    own_slow = (A.ms(att, replace(p, LOCK=False)) - A.ms(att, replace(p, LOCK=False, ignore_flags=True)))
    print(f"{tag}: r14c - F128 model {r['r14c'] - r['F128']:+.2f} (meas +2.08 .. +2.53); r14c - r14a {r['r14c'] - r['r14a']:+.2f}"
          f" (meas -0.03 .. -0.47); r14e - r14c {r['r14e'] - r['r14c']:+.2f} (meas -0.92 incl. ZASM -0.5? see note)")
    print(f"     no-lockstep: r14c - F128 {nl['r14c'] - nl['F128']:+.2f}; slow path own (cols independent) {own_slow:+.2f};"
          f" lockstep part of r14c {r['r14c'] - nl['r14c']:+.2f}, of F128 {r['F128'] - nl['F128']:+.2f}")
    return r


if __name__ == "__main__":
    raw = A.load()
    thin = A.load(0.109)
    p = A.P17()
    # own slow cycles, flat arithmetic
    fl = raw.fl[..., 1:22].mean()
    mv = raw.mv[..., 1:22].mean()
    units = 144 * 21 * 12 * 12
    own = units * (fl * (p.FP + p.T + p.XR) + mv * (p.R + p.MR)) / 24 / A.F / 1e3
    print(f"map: flagged {100 * fl:.2f} %, moved {100 * mv:.2f} % of chunk 1-21 units; own slow cycles {own:.2f} ms/step (raw)")
    flt = thin.fl[..., 1:22].mean()
    print(f"thin map: flagged {100 * flt:.2f} %")
    for name, att in (("raw", raw), ("thin", thin)):
        run(att, p, name)
        for d in (2, 4, 8):
            run(att, replace(p, D=d), f"{name} D={d}")
