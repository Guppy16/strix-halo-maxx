#!/bin/bash
# E7'' R = 66 / 67 (bank-3 cap hard), 4 workers; waits for solve_e7.sh to finish first
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
SHM_SCRATCH=$(cd "${SHM_SCRATCH:-$REPO/../.scratch}" && pwd)   # scratch / working dir (docker mounts it as /work)
cd "$SHM_SCRATCH/npu-research"
while pgrep -f "[s]olve_e7.sh" > /dev/null; do sleep 10; done
for spec in "66 8 900" "67 8 600"; do
  set -- $spec
  nice -n 10 ~/.local/bin/uv run --no-project --with ortools --with numpy python r22/gen_e7.py solve --R $1 --nx $2 \
    --time $3 --workers 4 --json r22/e7_r$1_x$2.json > r22/solve_r$1_x$2.log 2>&1
done
