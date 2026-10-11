"""price.py - whole-call static + in-situ (UNVERIFIED DMA-collision model, tools/ksplan.py) prices of the ROT variants
(research pass 14). Reads r19/rot_<v>.json (gen_rot fit) and replays the fitted .s for the per-bank busy fractions."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))
import gen_rot as G  # noqa: E402
import ksplan as K  # noqa: E402

TCALL = dict(g0f=1351, g1f=1368, g2f=1384, g0u=1366)  # tools/tcall.py r19/mm_rot_<v>.dis --func mm_rot
DEPLOYED_MEAS, DEPLOYED_STATIC, GAP_MEAS, GAP_P35 = 2122, 1824, 67, 55  # P35 (r18j) saved ~12 of the 67 (measured)
# round 18 KS-P38 (in situ, per k-step): MEMORY_STALL 173.7 vs the model's 221 for the deployed mm_ks -> scale 0.79; LOCK_STALL
# 13.3 (A waits) kept; DMAB (one core): +58-59 vs the model's 65 (0.9)
CAL, LOCK = 173.7 / 221, 13.3
f = K.MS_PER_CYCLE
print(f"{'variant':7s} {'tcall':>5s} {'b0':>5s} {'b1':>5s} {'b2':>5s} {'b3':>5s} {'coll':>5s} {'k-step':>6s} {'d ms':>6s} "
      f"{'+P35':>6s} {'no-coll':>7s} {'cal ks':>6s} {'cal d':>6s} {'+P35':>6s}")
for v, tc in TCALL.items():
    d = json.loads((HERE / f"rot_{v}.json").read_text())
    st = G.check((HERE / f"mm_rot_{v}.s").read_text(), d["info"], pad=0, order=[tuple(x) for x in d["order"]],
                 border=[tuple(x) for x in d["border"]])
    n = st["body_cycles"]
    busy = {b: st["busy"].get(b, 0) / n for b in range(4)}
    # A slot DMA writes: slot 0 in bank 1, slots 1 / 2 in bank 2 (r17d placement): 1/3 into bank 1, 2/3 into bank 2
    b12 = busy[1] / 3 + 2 * busy[2] / 3
    coll = K.collisions(None, busy[3], b12)
    ks = max(tc + coll, K.A_B / K.R_MT) + GAP_MEAS
    ks35 = max(tc + coll, K.A_B / K.R_MT) + GAP_P35
    ks0 = max(tc, K.A_B / K.R_MT) + GAP_MEAS
    ksc = max(tc + CAL * coll + LOCK, K.A_B / K.R_MT) + GAP_MEAS
    print(f"{v:7s} {tc:5d} {busy[0]:5.2f} {busy[1]:5.2f} {busy[2]:5.2f} {busy[3]:5.2f} {coll:5.0f} {ks:6.0f} "
          f"{(ks - DEPLOYED_MEAS) * f:+6.2f} {(ks35 - DEPLOYED_MEAS) * f:+6.2f} {(ks0 - DEPLOYED_MEAS) * f:+7.2f} "
          f"{ksc:6.0f} {(ksc - DEPLOYED_MEAS) * f:+6.2f} {(ksc - GAP_MEAS + GAP_P35 - DEPLOYED_MEAS) * f:+6.2f}")
print(f"core-busy fraction per bank over the 18 trips; coll = B prefetch 144 x b3 + A 324 x (b1 / 3 + 2 b2 / 3) [UNVERIFIED];")
print(f"k-step = max(static + coll, A port floor {K.A_B / K.R_MT:.0f}) + {GAP_MEAS} between calls ({GAP_P35} with P35); "
      f"no-coll = the DMA yields (DMAB outcome 2)")
print(f"cal = calibrated on round 18's KS-P38: coll x {CAL:.2f} + LOCK {LOCK} (MEMORY_STALL 173.7 measured vs 221 modelled)")
print(f"deployed: static {DEPLOYED_STATIC}, measured k-step {DEPLOYED_MEAS}; 1 cycle per k-step = {f * 1e3:.2f} us per step")
