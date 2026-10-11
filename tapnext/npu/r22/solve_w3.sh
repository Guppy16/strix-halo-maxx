#!/bin/bash
# round 22: E7 R / nx with the bank-3 excess weighed as its real cost (18 stalls per call per excess residue = E7_W3 1800 against
# 100 per modelled stall; research's 10000 traded ~5 stalls of other conflicts for one stall of bank 3), repaired from a hint.
# usage: r22/solve_w3.sh R NX N1 HINT TIME TAG
cd ~/git/app/.scratch/npu-tapnext
R=$1; NX=$2; N1=$3; H=$4; TM=$5; TAG=$6
E7_W3=${E7_W3:-1800} E7_DROP=bank3soft nice -n 10 ~/.local/bin/uv run --no-project --with ortools --with numpy python r22/gen_e7m.py solve \
  --R $R --nx $NX --n1 $N1 --time $TM --workers ${W:-4} --hint $H --json r22/e7_$TAG.json > r22/solve_$TAG.log 2>&1
