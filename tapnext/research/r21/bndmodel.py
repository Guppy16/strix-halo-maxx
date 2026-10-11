"""Rev 16: the op-boundary model, checked against main's round-20 TPROF dump (r20/tprof20.py npz: per tile first / ks / nks /
mid / epi / hand / t0 / tile counter; the 24 GEMM cores' timers agree within ~1k cycles) and the CPST probe (round 20: the CP
passes mlp3_a's dependency wait 15.84 us after mlp0's last k-step end, 14.88 after the first of its 8 tct, 15.37 after the 4th;
core 0_2's first k-step ends 19.99 us after, the median core's 22.71).

Boundary of a dependent GEMM (its A = the previous op's output, through DDR):
  tail   producer's last epilogue end -> the CP passes the first tct of the wait   (CPST: 12.2 us after mlp0's gelu_pan epilogue)
  tct    the 8-tct run                                                            (CPST: ~1.0 us)
  issue  CP after the wait: 8 A fills (BD write + patch + maskpoll + push) then the memtile sends column by column
         (push = maskpoll + write32 = 0.17 us, CPRATE); column c's t0 send is pushed after 8 fills + c x n_send pushes
  first  chunk 0 -> memtile -> core + one k-step
  pace   the first tile's k-steps paced by the A load at the DDR read cap (all 8 columns load at once, plus B if not prefetched)
Independent ops (A preloaded during the previous op's last tile) pay none of these except a residual pace.
usage: python3 bndmodel.py tp_r20t2.npz [block]"""
import sys

import numpy as np

G = 1810.0
d = np.load(sys.argv[1])
a, names = d["a"], d["names"]
blk = sys.argv[2] if len(sys.argv) > 2 else "b11"
C = dict(op=0, col=1, byte=2, first=3, ks=4, nks=5, mid=6, epi=7, hand=8, t0=9, n=10)
CAP = 57.0  # GB/s (DDRCAP 55-59)
T_PUSH = 0.17  # us (maskpoll 0.11 + write32 0.06, CPRATE)

ops = []
for op in sorted(set(a[:, 0])):
    m = a[:, 0] == op
    if names[m][0].startswith(blk + "_"):
        ops.append((names[m][0].split("_", 1)[1], a[m]))
# per-op facts: K, A bytes over 8 columns, sends per column, dependent on the previous GEMM-core op?
FACTS = dict(out=(768, 995328, 3, "ssm"), up_a=(768, 995328, 2, "norm1"), up_b=(768, 0, 2, None), down_a=(1536, 1990656, 5, None),
             down_b=(1536, 1990656, 5, None), qkv=(768, 995328, 3, "norm2"), out_proj=(768, 995328, 3, "attn"),
             mlp0=(768, 995328, 3, "norm3"), mlp3_a=(1536, 1990656, 5, "mlp0"), mlp3_b=(1536, 1990656, 5, None))
prev_end = None
print(f"block {blk}; us; W = previous GEMM-core op's last epilogue end -> this op's median first-k-step end")
print("op        prev     W_meas  ks1/k  pace_meas  pace_cap(A only)  first-tile compute")
rows = []
for nm, r in ops:
    end = r[:, C["t0"]] + r[:, C["first"]] + r[:, C["ks"]] + r[:, C["mid"]] + r[:, C["epi"]]
    n0 = r[:, C["n"]].min()
    f = r[r[:, C["n"]] == n0]
    nks = int(np.median(f[:, C["nks"]]))
    ksk = np.median(f[:, C["ks"]] / np.maximum(f[:, C["nks"]] - 1, 1))
    pace = (ksk - 1576) * (nks - 1) / G
    k0e = np.median(f[:, C["t0"]] + f[:, C["first"]])
    W = (k0e - prev_end) / G if prev_end is not None else float("nan")
    fx = FACTS.get(nm)
    if fx:
        K, abytes, ns, dep = fx
        comp = K / 64 * 1576 / G
        pcap = max(0.0, abytes / CAP / 1e3 - comp)
        print(f"{nm:9s} {str(dep):7s} {W:7.2f} {ksk:6.0f} {pace:9.2f} {pcap:13.2f} {comp:12.2f}")
        rows.append((nm, dep, W, pace))
    prev_end = np.max(end)
print()
tail, tct = 12.2, 1.0
print("mlp3_a (GEMM -> GEMM, CPST-anchored) decomposition, us after mlp0's last epilogue end:")
issue0 = 8 * (0.10 + T_PUSH) + 2 * T_PUSH
print(f"  tail {tail} + tct {tct} + issue to column 0's sends {issue0:.2f} (BD write + patch ~0.10 us INFERRED) + chunk 0 + k-step"
      f" ~1.6 = {tail + tct + issue0 + 1.6:.1f} (CPST core 0_2: 19.99 - 2.71 = 17.3)")
print(f"  per column +5 sends x {T_PUSH} = {5 * T_PUSH:.2f} us: median core (column ~3.5) +{3.5 * 5 * T_PUSH:.1f} (CPST: +2.7)")
pace_sum = sum(max(p, 0) for _, _, _, p in rows)
print(f"\nfirst-tile A pacing, sum over the block's GEMMs: {pace_sum:.1f} us -> x12 = {12 * pace_sum / 1e3:.2f} ms per step")
