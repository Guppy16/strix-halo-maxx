"""Print A W C buffer sizes in bytes for a design build (for host.cpp)."""

import json
import sys

from design import plan

cfg = json.load(open(sys.argv[1] + ".json"))
ops, a_tot, w_tot, c_tot = plan(json.loads(cfg["sched"]), cfg["m"], cfg["k"], cfg["n"], cfg["rho"])
print(a_tot * 4, w_tot * cfg["reps"] * 4, c_tot * 2)
