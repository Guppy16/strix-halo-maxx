#!/bin/bash
# round 17: npu-research/tools/try.sh on the current best (full12r16z = r15m + SQK INPART_INLINE XONE GGK SSQ SIG10 N1K, -SSX).
# usage: r17/try17.sh NAME [try.sh options]      (BASE_MS overrides the baseline the delta is printed against)
name=$1; shift
exec ~/git/app/.scratch/npu-research/tools/try.sh "$name" --ew-add "-DSQK" --ew-add "-DINPART_INLINE" --ew-add "-DXONE" \
  --ew-add "-DGGK" --ew-del "-DSSX" --ew-add "-DSSQ" --ew-add "-DSIG10" --ew-add "-DN1K" --base-ms ${BASE_MS:-39.94} "$@"
