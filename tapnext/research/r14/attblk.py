"""rev 9: per-block check of attsim against the host-polled timelines. The attention wall varies 744-920 us per block in r13f;
the map's flag rates vary by block too (b10 / b11 almost none). Model per block (mean of the 3 map frames) vs measured walls
(blocks 1-11; poll quantum ~17 us), for r13f (flag), r13b (exact test on every unit) and r12e (C++ chunk_k: the exp on every
unit, the rescale where the max moves).
    python3 attblk.py"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import attsim as A  # noqa: E402

HERE = Path(__file__).resolve().parent
TLB = HERE.parents[1] / "npu-tapnext" / "build"


def attn_walls(tag):  # as tools/tlops.walls: last(op) - last(previous op), blocks 1-11
    d = json.loads((TLB / f"tl_full12{tag}_d12.json").read_text())
    w, prev = [], None
    for o in d["ops"]:
        if o["last"] is None:
            continue
        blk, _, kind = o["name"].partition("_")
        if prev is not None and blk != "b0" and kind == "attn":
            w.append((o["last"] - prev) * 1e3)
        prev = o["last"]
    return np.array(w)
pb = replace(A.Prm(**json.load(open(HERE / "attcal.json"))), D=8)
att = A.default_att().thinned(float(sys.argv[1])) if len(sys.argv) > 1 else A.default_att()
if len(sys.argv) > 2:
    pb = replace(pb, R=float(sys.argv[2]))
for tag, v, prm, note in (("r13f", dict(a=1, test="flag", exp="flag"), pb, "deployed"),
                          ("r13b", dict(a=1, test="all", exp="move"), replace(pb, D=4), "test on every unit"),
                          ("r12e", dict(a=1, test="all", exp="all"), replace(pb, D=4, L=571 - 66 - 400 - 68 + 400), "C++ chunk_k (approx.)")):
    t_end, _ = att.block_cycles(att.counts(v), prm, v, detail=True)
    mod = t_end.reshape(-1, 12).mean(0) / A.F  # us per block 0..11
    meas = attn_walls(tag)[:11]
    m = mod[1:12]
    dm, dt = m - m.mean(), meas - meas.mean()
    slope = float((dm * dt).sum() / (dm * dm).sum())
    r = float(np.corrcoef(dm, dt)[0, 1])
    print(f"{tag} ({note}): corr {r:+.2f}, slope measured/model {slope:.2f}; spread model {m.max() - m.min():.0f} us, "
          f"measured {meas.max() - meas.min():.0f} us")
    print("   model " + " ".join(f"{x:5.0f}" for x in m))
    print("   meas  " + " ".join(f"{x:5.0f}" for x in meas))
