"""rev 10: per-block check of sim17 against the r14 timelines (attention wall per block, blocks 1-11), kappa 0.5 / 0.6 / 0.7.
    python3 blk17.py"""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

import attsim17 as A
from fit17 import kappa_att

TLB = Path(__file__).resolve().parents[2] / "npu" / "build"


def walls(tag):
    d = json.loads((TLB / f"tl_full12{tag}_d12.json").read_text())
    w, prev = [], None
    for o in d["ops"]:
        if o["last"] is None:
            continue
        blk, _, kind = o["name"].partition("_")
        if prev is not None and blk != "b0" and kind == "attn":
            w.append((o["last"] - prev) * 1e3)
        prev = o["last"]
    return np.array(w[:11])


if __name__ == "__main__":
    raw = A.load()
    P = A.P17()
    builds = {"r14a": replace(P, batched=False), "r14c": P, "r14f": replace(P, VRB=136.0)}
    for kappa in (0.5, 0.6, 0.7, 1.0):
        for tag, p in builds.items():
            mods = []
            for sd in (0, 1, 2, 3):
                t_end, _ = kappa_att(raw, kappa, sd).sim(p, detail=True)
                mods.append(t_end.reshape(-1, 12).mean(0) / A.F)
            m = np.mean(mods, 0)[1:12]
            meas = walls(tag)
            dm, dt = m - m.mean(), meas - meas.mean()
            slope = float((dm * dt).sum() / (dm * dm).sum())
            r = float(np.corrcoef(dm, dt)[0, 1])
            print(f"kappa {kappa} {tag}: corr {r:+.2f} slope meas/model {slope:.2f}; mean model {m.mean():.0f} meas {meas.mean():.0f} us; "
                  f"spread model {m.max() - m.min():.0f} meas {meas.max() - meas.min():.0f}")
            if kappa == 0.6:
                print("    model " + " ".join(f"{x:4.0f}" for x in m))
                print("    meas  " + " ".join(f"{x:4.0f}" for x in meas))
