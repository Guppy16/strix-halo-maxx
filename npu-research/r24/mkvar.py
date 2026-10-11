"""pass 19 KTLP-NL: lock-free, instrumented variants of the k-step kernels for a one-core bench that cannot hang.
Every lock op keeps its slot and bundle (the schedule is unchanged) but becomes an alu add that accumulates its condition:
  KTL (mm_e7k -> mm_e7k_nl): acq.cond #53 -> add r8 (A acquires), rel.cond #52 -> add r9 (A releases), acq.cond #49 -> add r10
  (B acquires), rel.cond #48 -> add r11 (B releases); setup saves r8-r11, copies the 3rd pointer argument (p2, the log) to p5 before
  the kernel overwrites p2, zeroes the counters; after the loop it logs r8-r11, r16-r19 (the buffer bases), r27 (counter), r6
  (MAC config), p2 (record pointer) and restores r8-r11. Signature: mm_e7k_nl(rec, pc', log, cfg, trips, y, x).
  E7 (mm_e7 -> mm_e7_nl): lock ops -> add r28, r28, r26 (no locks, nothing logged). Signature unchanged.
usage: python3 r24/mkvar.py ktl IN.s OUT.s | e7 IN.s OUT.s"""
import re
import sys

kind, src, dst = sys.argv[1:4]
s = open(src).read()
if kind == "ktl":
    reps = {"acq.cond #53, r4, r26": "add r8, r8, r26", "rel.cond #52, r5, r26": "add r9, r9, r26",
            "acq.cond #49, r4, r26": "add r10, r10, r26", "rel.cond #48, r5, r26": "add r11, r11, r26"}
    for a, b in reps.items():
        n = s.count(a)
        assert n >= 1, a
        s = s.replace(a, b)
    s = s.replace("mm_e7k", "mm_e7k_nl")
    head = ["movs p5, p2", "paddxm [sp], #64", "nop", "st r8, [sp, #-64]", "st r9, [sp, #-60]", "st r10, [sp, #-56]",
            "st r11, [sp, #-52]", "mova r8, #0", "mova r9, #0", "mova r10, #0", "mova r11, #0"]
    s = s.replace("mm_e7k_nl:\n", "mm_e7k_nl:\n" + "".join(f"\t{x}\n" for x in head), 1)
    log = ["nop"] * 8 + [f"st {r}, [p5], #4" for r in ("r8", "r9", "r10", "r11", "r16", "r17", "r18", "r19", "r27", "r6")] + \
          ["mov r0, p2", "nop", "st r0, [p5], #4", "lda r8, [sp, #-64]", "lda r9, [sp, #-60]", "lda r10, [sp, #-56]",
           "lda r11, [sp, #-52]", "paddxm [sp], #-64"] + ["nop"] * 4
    i = s.index("\tret lr")
    s = s[:i] + "".join(f"\t{x}\n" for x in log) + s[i:]
else:
    for a in ("acq.cond #53, r4, r26", "rel.cond #52, r5, r26"):
        assert a in s, a
        s = s.replace(a, "add r28, r28, r26")
    s = re.sub(r"mm_e7(?=[_:,\s@-]|$)", "mm_e7_nl", s, flags=re.M)
open(dst, "w").write(s)
print(kind, src, "->", dst)
