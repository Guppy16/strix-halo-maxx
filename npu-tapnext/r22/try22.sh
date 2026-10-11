#!/bin/bash
# round 22: r21/try21.sh (full12r21k1) + the round-22 kept changes: A3 (the third A slot region for down_b / mlp3_b; A3=0 without).
# usage: r22/try22.sh NAME [--sweep] [--ew-add ..] [--k-add ..] [--env K=V] [--args ..]   (NOREG / NOTIME / E7_V / E7_X0 .. as try21)
name=$1; shift
cd ~/git/app/.scratch/npu-tapnext || exit 1
R22=()
[ "${A3:-1}" = 1 ] && R22+=(--env A3=1)
exec r21/try21.sh "$name" "${R22[@]}" "$@"
