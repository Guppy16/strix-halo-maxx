#!/bin/bash
# round 22: E7 at nx = 10, n1 = 30 (ny = 24 keeps the lf1 segment start 64 B aligned; bank 3 carries 61 reads per row block, not 63)
# repaired from research pass 17 nx = 8 solutions. usage: r22/solve_nx10.sh R HINT TIME [NX]
cd ~/git/app/.scratch/npu-tapnext
R=$1; H=$2; TM=$3; NX=${4:-10}
E7_DROP=bank3soft nice -n 10 ~/.local/bin/uv run --no-project --with ortools --with numpy python ../npu-research/r22/gen_e7.py solve \
  --R $R --nx $NX --n1 ${N1:-30} --time $TM --workers 4 --hint $H --json r22/e7_r${R}_x${NX}.json > r22/solve_r${R}_x${NX}.log 2>&1
