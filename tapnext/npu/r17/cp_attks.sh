#!/bin/bash
# round 17 (host): CP-solve the S2 attkb schedule; usage: r17/cp_attks.sh TAG [gen_attkb.py args]
# (all compute on the test machine: nice -n 10, at most 12 CP workers - the box is shared)
tag=$1; shift
cd "$(dirname "$0")/.." && mkdir -p build/r17 && nice -n 10 ~/.local/bin/uv run --no-project --with ortools --with numpy python r14/gen_attkb.py build/r17/attks_$tag.s --s2 --cp r17/attks_$tag.json --workers 12 "$@" > build/r17/attks_${tag}_cp.out 2>&1
