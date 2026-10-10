"""round 16: gen_sqk.table()'s t equals the deployed ssqrt_ab / ssqrt_cd (r9/ssm_lut.h)"""
import re
import sys

import numpy as np

sys.path.insert(0, "r16")
import gen_sqk  # noqa: E402

t, tp = gen_sqk.table()
txt = open("r9/ssm_lut.h").read()
for nm in ("ssqrt_ab", "ssqrt_cd"):
    m = re.search(nm + r"\[512\] = \{(.*?)\};", txt, re.S)
    orig = np.array([int(x) for x in m.group(1).replace("\n", " ").split(",") if x.strip()], np.int32)
    g = t.astype(np.int16).reshape(-1, 1, 16)
    lay = np.concatenate([g, g], 1).reshape(-1)
    print(nm, len(orig), bool((orig == lay).all()), orig[:3], tp[:3], tp.min(), tp.max())
