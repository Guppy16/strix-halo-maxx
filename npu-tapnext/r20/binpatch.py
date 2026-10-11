"""Round 20: post-process a build's runtime sequences (.bin = the CP's TXN stream) and save them as a new build name.

Copies build/full<d>SRC{,_p<k>}.{bin,sched,json,xclbin} to ...DST and rewrites each .bin with a transformation:
  cpst=K     move the CPST probe's flag write (core 0_2, CPST_ADDR) to just after the K-th tct of the run before it
usage (npu-iron): python3 r20/binpatch.py SRC DST RULE [--depths 12]
"""
import argparse
import os
import shutil
import sys

import numpy as np
from aie.utils import txn_trace as T

B = "/work/npu-tapnext/build"


def serialize(ops):
    words = [w for o in ops[1:] for w in o.words]
    h = list(ops[0].words)
    h[2] = len(ops) - 1
    h[3] = 4 * (4 + len(words))
    return np.array(h + words, np.uint32)


def rule_cpst(ops, k, addr):
    i = next(j for j, o in enumerate(ops) if o.kind == "write32" and o.addr == addr and o.value == 1)
    j = i - 1
    while ops[j].kind != "tct":
        j -= 1
    end = j
    while ops[j - 1].kind == "tct":
        j -= 1
    assert end - j + 1 >= k, (end - j + 1, k)
    w = ops.pop(i)
    ops.insert(j + k, w)
    print(f"  cpst: flag write moved from {i} to after tct {k} of {end - j + 1}")
    return ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("rule")
    ap.add_argument("--depths", default="12")
    o = ap.parse_args()
    for d in map(int, o.depths.split(",")):
        for ph in ("", "_p1", "_p2", "_p3"):
            s, t = f"{B}/full{d}{o.src}{ph}", f"{B}/full{d}{o.dst}{ph}"
            if not os.path.exists(s + ".bin"):
                continue
            for ext in (".sched", ".sched.meta", ".json", ".xclbin"):
                if os.path.exists(s + ext):
                    shutil.copy(s + ext, t + ext)
            ops = T.decode(T.load(s + ".bin"))
            name, _, arg = o.rule.partition("=")
            if name == "cpst":
                ops = rule_cpst(ops, int(arg), (2 << 20) | 0x7B40)
            else:
                sys.exit(f"unknown rule {name}")
            out = serialize(ops)
            T.decode(out)
            out.tofile(t + ".bin")
            print(f"{t}.bin: {len(ops) - 1} ops")
        os.system(f"chmod a+rw {B}/full{d}{o.dst}*")


if __name__ == "__main__":
    main()
