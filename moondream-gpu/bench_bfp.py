"""bench.py with the bfp16 emulation classes available to --vlin / --tlin."""

import runpy
import sys

import bfpemu
import md2fast

for name in ("BfpLin", "BfpWLin", "BfpWAcc32Lin", "AccLin", "BfpAcc32Lin", "BfpRotLin"):
    setattr(md2fast, name, getattr(bfpemu, name))
sys.argv = ["bench.py", *sys.argv[1:]]
runpy.run_path("bench.py", run_name="__main__")
