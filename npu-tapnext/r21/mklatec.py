"""Round 21 LATEC: from a ZLOAD ROT kernel (r21/mkzload.py) derive the late-C variant: the per-record A-slot acquire
`acq.cond #53, r4, r26` becomes `acq.cond r26, r4, r26`, i.e. the record's acquire field holds the LOCK ID (53 = the A slot's full lock,
48 + 5; 0 = none). mm8.cc -DLATEC writes the C sub-part locks (48 + CS_LB / CS_LC) into the odd row blocks 3 / 5 of a fresh object's
first k-step: the k-step starts once sub-part 1 is drained and waits for sub-parts 2 / 3 only before storing into them.
usage: python3 r21/mklatec.py r21/mm_rot_g0z.s r21/mm_rot_g0l.s"""
import sys

src = open(sys.argv[1]).read()
n = src.count("acq.cond #53, r4, r26")
assert n == 2, n
open(sys.argv[2], "w").write(src.replace("acq.cond #53, r4, r26", "acq.cond r26, r4, r26"))
print(f"mklatec: {n} acquires -> register lock id")
