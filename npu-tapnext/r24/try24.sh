#!/bin/bash
# round 24: r23/try23.sh (full12r23z) + the round-24 kept changes:
#   STATS64 (the norm statistics of both 32-row halves as one 64-lane chain, LN and RMS; STATS64=0 without).
# Dead / off by default (env switches kept in design8 / plan8): MSDEFER (out's pass sends after yx's drains, with MSPLIT incl. out), YXH
# (yx steps head-major; drains in 4 groups).
# usage: r24/try24.sh NAME [--sweep] [--ew-add ..] [--k-add ..] [--env K=V] [--args ..]   (NOREG / NOTIME / MSPLIT / YXSSM / .. as try23)
name=$1; shift
cd ~/git/app/.scratch/npu-tapnext || exit 1
R24=()
[ "${STATS64:-1}" = 1 ] && R24+=(--ew-add -DSTATS64)
exec r23/try23.sh "$name" "${R24[@]}" "$@"
