#!/bin/bash
# round 22: r21/try21.sh (full12r21k1) + the round-22 kept changes: A3 (the third A slot region for down_b / mlp3_b; A3=0 without),
# TXT18 (EW text -448 B; TXT18=0 without), attkb at II 34 (ATTKS_JSON=r17/attks.json for the old II 36).
# usage: r22/try22.sh NAME [--sweep] [--ew-add ..] [--k-add ..] [--env K=V] [--args ..]   (NOREG / NOTIME / E7_V / E7_X0 .. as try21)
name=$1; shift
cd ~/git/app/.scratch/npu-tapnext || exit 1
R22=()
[ "${A3:-1}" = 1 ] && R22+=(--env A3=1)
R22+=(--env ATTKS_JSON=${ATTKS_JSON:-r22/attks34.json})  # round 22: attkb (S2) re-solved at II 34 (r17: 36)
[ "${TXT18:-1}" = 1 ] && R22+=(--ew-add -DTXT18)  # research pass 18: EW text (norm twins by pointer, stats loop, one ring copy)
exec r21/try21.sh "$name" "${R22[@]}" "$@"
