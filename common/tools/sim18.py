"""sim18 (npu-research rev 11): the attention slow-path levers on the deployed full12r15m (41.52 ms at depth 12).

attsim17 (multicast lockstep + head-group barrier, per-row-block one-core costs, kappa 0.5 / 0.6 / 0.7) with the round-15
state: P60g applied (measured -1.06 vs -1.08 predicted) and S1 (resc) at the rescale cost R that reproduces its MEASURED
-0.29 ms (fitted here, kappa 0.6). Then S2 in its rev-11 form (768 B of max slots on attkb's stack frame, bank 0; II 34 -> 35
budget, B +7 per row block; 36 as the pessimistic case), S3, V1, K1, K2 cumulative.
    OPENBLAS_NUM_THREADS=1 python3 sim18.py      (numpy, ~5 min)
"""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attsim17 as A  # noqa: E402
import sim17 as S  # noqa: E402


def main():
    raw = A.load()
    P = replace(A.P17(), VRB=136.0)
    G = S.g6()
    atts = {k: [S.kappa_att(raw, k, s) for s in range(4)] for k in (0.5, 0.6, 0.7)}

    def ms(k, **o):
        return np.mean([A.ms(A.remap(a, G), replace(P, **o)) for a in atts[k]])
    # S1 calibration: R such that ms(R) - ms(578) = -0.29 at kappa 0.6 (bisection)
    b6 = ms(0.6)
    lo, hi = 150.0, 578.0
    for _ in range(14):
        mid = (lo + hi) / 2
        if ms(0.6, R=mid) - b6 < -0.29:
            lo = mid
        else:
            hi = mid
    R1 = (lo + hi) / 2
    print(f"S1 calibration: R = {R1:.0f} cycles in situ reproduces the measured -0.29 ms (rev 10 assumed 200)")
    rows = [("S2: running max on the stack, II 35 (B 268)", dict(R=R1, FP=0.0, T=70.0, B=268.0)),
            ("S2 pessimistic: II 36 (B 275), T 100", dict(R=R1, FP=0.0, T=100.0, B=275.0)),
            ("S2 optimistic: II 34 kept (B 261)", dict(R=R1, FP=0.0, T=70.0, B=261.0)),
            ("+ S3 test + rescale as one asm call (after S2 II 35)", dict(R=R1 - 30, FP=0.0, T=40.0, B=268.0)),
            ("+ V1 row sums folded into attv", dict(R=R1 - 30, FP=0.0, T=40.0, B=268.0, VRB=110.0)),
            ("+ K1 batched prep / flag scan in asm", dict(R=R1 - 30, FP=0.0, T=40.0, B=237.0, VRB=110.0)),
            ("+ K2 chunk 0 QK-only + test", dict(R=R1 - 30, FP=0.0, T=40.0, B=237.0, VRB=110.0, C0=600.0)),
            ("floor: no slow path at all (flags ignored), r15m costs", dict(R=R1, ignore_flags=True))]
    print("ms per step vs r15m (P60g + S1 measured); kappa 0.5 / 0.6 / 0.7")
    for name, o in rows:
        d = [ms(k, **o) - ms(k, R=R1) for k in (0.5, 0.6, 0.7)]
        print(f"  {d[0]:+6.2f} / {d[1]:+6.2f} / {d[2]:+6.2f}  {name}")


if __name__ == "__main__":
    main()
