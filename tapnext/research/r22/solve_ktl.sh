#!/bin/bash
# KTL feasibility: every stream segment instance serves one row block (--aligned), lf1 split L | N (N serves the next row block)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)/setup.sh"
cd "$SHM_SCRATCH/npu-research"
for spec in "68 16" "68 8" "70 16"; do
  set -- $spec
  nice -n 10 ~/.local/bin/uv run --no-project --with ortools --with numpy python r22/gen_e7.py solve --R $1 --nx 8 --n1b $2 --aligned \
    --time 300 --workers 4 --json r22/e7a_r$1_n$2.json > r22/solve_a_r$1_n$2.log 2>&1
  cat r22/solve_a_r$1_n$2.log
done
