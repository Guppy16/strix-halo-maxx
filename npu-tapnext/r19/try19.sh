#!/bin/bash
# round 19: r18/try18.sh on the round-19 best (r18m + ROT g0a). usage: r19/try19.sh NAME [--sweep] [--ew-add ..] [--k-add ..] [--env K=V]
name=$1; shift
exec ~/git/app/.scratch/npu-tapnext/r18/try18.sh "$name" --env ROT_V=g0a "$@"
