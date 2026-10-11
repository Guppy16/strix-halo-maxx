#!/bin/bash
# R = 64 feasibility diagnostics in residue mode (6 workers): which constraint group binds
cd ~/git/app/.scratch/npu-research
for d in "soft,nb,bank3,bank0,qey:240" "soft,nb:300" "soft,bank3:300" "soft:400"; do
  dd=${d%%:*}; tt=${d##*:}
  echo "== drop $dd ($tt s)"
  E7_DROP=$dd nice -n 10 ~/.local/bin/uv run --no-project --with ortools --with numpy python r22/gen_e7.py solve --R 64 --time $tt --workers 6 --json r22/diag64_${dd//,/_}.json 2>&1 | tail -1
done
