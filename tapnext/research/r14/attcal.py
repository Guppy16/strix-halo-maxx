"""npu-research rev 9: calibrate tools/attsim.py on the round-13 twins (F3 / F2 / F16 / r13c / F8 at l2_depth 4) and check the
holdouts r13e (l2_depth 8) and r13f (+ the in-loop flag, the deployed build).
    python3 attcal.py
"""
import itertools
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common" / "tools"))
import attsim as A  # noqa: E402

att = A.default_att()
cnts = {k: att.counts(v) for k, (v, _, _) in A.V_TWINS.items()}
FIT = ["F3", "F2", "F16", "r13c", "F8"]
meas = {k: m for k, (_, _, m) in A.V_TWINS.items()}


def evaluate(prm):
    raw = {k: 12 * att.block_cycles(cnts[k], replace(prm, D=A.V_TWINS[k][1])) / A.F / 1e3 for k in A.V_TWINS}
    base = np.mean([meas[k] - raw[k] for k in FIT])
    pred = {k: base + raw[k] for k in raw}
    rms = np.sqrt(np.mean([(pred[k] - meas[k]) ** 2 for k in FIT]))
    return rms, base, pred, raw


def show(tag, prm):
    rms, base, pred, raw = evaluate(prm)
    print(f"{tag}: rms {rms:.3f} ms (fit set), base {base:.2f} ms; L {prm.L:.0f} T {prm.T:.0f} R {prm.R:.0f} E {prm.E:.0f} "
          f"P_SHIM {prm.P_SHIM:.0f}")
    for k in A.V_TWINS:
        tag2 = "" if k in FIT else "  (holdout)"
        print(f"    {k:5s} measured {meas[k]:6.2f}  model {pred[k]:6.2f}  err {pred[k] - meas[k]:+5.2f}  attention "
              f"{raw[k] / 12 * 1e3:6.0f} us/block{tag2}")
    return rms


p0 = A.Prm()
print("prior parameters (bench / stamps / twins as read):")
show("prior", p0)

# coordinate grid on the four unknowns that matter (L, T, R, E) and the producer rate
best = (9e9, None)
for ps in (960, 1280, 1600, 2000):
    for L in range(300, 421, 20):
        for E in (360, 400, 440):
            for T in (66, 100, 130):
                for R in (100, 300, 580):
                    prm = replace(p0, L=L, E=E, T=T, R=R, P_SHIM=ps)
                    r = evaluate(prm)[0]
                    if r < best[0]:
                        best = (r, prm)
print()
show("best grid fit", best[1])
pb = best[1]
# local refinement on L, T, E
for it in range(2):
    for name, step in (("L", 5), ("T", 8), ("E", 8), ("R", 50), ("P_SHIM", 80)):
        cur = getattr(pb, name)
        cands = [replace(pb, **{name: cur + k * step}) for k in (-2, -1, 0, 1, 2)]
        pb = min(cands, key=lambda q: evaluate(q)[0])
print()
show("refined", pb)

print("\nlockstep counterfactual (same parameters, columns independent):")
for k in ("F2", "r13c", "r13f"):
    v, d, _ = A.V_TWINS[k]
    a = att.block_cycles(cnts[k], replace(pb, D=d))
    b = att.block_cycles(cnts[k], replace(pb, D=d, LOCK=False))
    print(f"    {k:5s} lockstep {a / A.F:7.1f} us/block, independent {b / A.F:7.1f}: lockstep costs {12 * (a - b) / A.F / 1e3:.2f} ms/step")
t_end, wait = att.block_cycles(cnts["r13f"], pb, detail=True)
print(f"r13f: mean core wait for K / V objects {wait.mean() / A.F:.1f} us per block ({12 * wait.mean() / A.F / 1e3:.2f} ms/step)")
for D in (2, 4, 6, 8, 12, 16):
    print(f"    r13f at l2_depth {D:2d}: {12 * att.block_cycles(cnts['r13f'], replace(pb, D=D)) / A.F / 1e3:.2f} ms attention/step")
for ns in (2, 3, 4):
    print(f"    r13f with {ns} L1 K / V slots: {12 * att.block_cycles(cnts['r13f'], replace(pb, SLOTS=ns)) / A.F / 1e3:.2f} ms")
import json  # noqa: E402
(Path(__file__).resolve().parent / "attcal.json").write_text(json.dumps(pb.__dict__))
