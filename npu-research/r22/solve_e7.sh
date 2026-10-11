#!/bin/bash
# E7'' CP-SAT, residue mode, 4 workers (main measured the NPU clock dropping to 1.4 GHz under ~20 busy CPU threads, round 21):
# the bank-3 cap as a cost, started from the relaxed (no bank-3 cap) R = 64 solution
cd ~/git/app/.scratch/npu-research
H=r22/diag64_soft_bank3.json
for spec in "64 8 1200" "65 8 900"; do
  set -- $spec
  E7_DROP=bank3soft nice -n 10 ~/.local/bin/uv run --no-project --with ortools --with numpy python r22/gen_e7.py solve --R $1 --nx $2 \
    --time $3 --workers 4 --hint $H --json r22/e7_r$1_x$2.json > r22/solve_r$1_x$2.log 2>&1
done
