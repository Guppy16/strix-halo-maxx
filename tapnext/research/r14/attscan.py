"""rev 9: sensitivity of the attsim fit (T, R, L) on the round-13 twins.  python3 attscan.py"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common" / "tools"))
import attsim as A  # noqa: E402

att = A.default_att()
cnts = {k: att.counts(v) for k, (v, _, _) in A.V_TWINS.items()}
pb = A.Prm(**json.load(open(Path(__file__).resolve().parent / "attcal.json")))
FIT = ["F3", "F2", "F16", "r13c", "F8"]
meas = {k: m for k, (_, _, m) in A.V_TWINS.items()}


def ev(p):
    raw = {k: 12 * att.block_cycles(cnts[k], replace(p, D=A.V_TWINS[k][1])) / A.F / 1e3 for k in A.V_TWINS}
    base = np.mean([meas[k] - raw[k] for k in FIT])
    pred = {k: base + raw[k] for k in raw}
    return np.sqrt(np.mean([(pred[k] - meas[k]) ** 2 for k in FIT])), pred


RS = (0, 100, 200, 300, 400, 600)
print("rms (ms) over the fit set; rows T, columns R", RS)
for T in (30, 66, 100, 130, 160):
    print(f"  T {T:3d}: " + " ".join(f"{ev(replace(pb, T=T, R=R))[0]:.3f}" for R in RS))
for L in (380, 390, 400, 410, 420):
    r, p = ev(replace(pb, L=L))
    print(f"  L {L}: rms {r:.3f}, r13f model {p['r13f']:.2f} (measured 49.21)")
for ps in (1200, 1400, 1600, 1800, 2000):
    r, p = ev(replace(pb, P_SHIM=ps))
    print(f"  P_SHIM {ps}: rms {r:.3f}, F3 {p['F3']:.2f} (43.39)")
