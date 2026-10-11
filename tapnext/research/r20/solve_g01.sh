#!/bin/bash
# pass 15: re-solve ROT g1f / g0f with the round-18 lock gaps (ACQ_GAP 6, REL_GAP 8 = hard floors in r20/gen_rot.py) on the
# full12r18m placement (+ acc_hi drift of +-64 B, which KSC moves whenever .data grows)
cd "$(dirname "$0")/.."
P="c_obj=0x78900,slots=0x77b80:0x78000:0x78480,rec=0x71804,b=0x7d100"
UV="nice -n 10 $HOME/.local/bin/uv run --no-project --with ortools --with numpy python"
for v in "$@"; do
  g=${v:1:1}
  $UV r20/gen_rot.py solve --g "$g" --free --time 900 --workers 12 --json "r20/rot_$v.json" \
    --place "$P,acc_hi=0x71980" --place "$P,acc_hi=0x71940" --place "$P,acc_hi=0x719c0" > "r20/solve_$v.log" 2>&1
  echo "$v rc=$?" >> r20/solve_g01.done
done
