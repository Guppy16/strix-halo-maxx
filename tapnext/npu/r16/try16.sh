#!/bin/bash
SHM_REPO=$(cd "$(dirname "$0")" && git rev-parse --show-toplevel); SHM_SCRATCH=${SHM_SCRATCH:-$SHM_REPO/../.scratch}
# round 16: common/tools/try.sh on the current best (full12r16e = r15m + SQK INPART_INLINE XONE GGK SSQ SIG10 N1K, -SSX).
# usage: r16/try16.sh NAME [try.sh options]      (BASE_MS overrides the baseline the delta is printed against)
name=$1; shift
exec $SHM_REPO/common/tools/try.sh "$name" --ew-add "-DSQK" --ew-add "-DINPART_INLINE" --ew-add "-DXONE" \
  --ew-add "-DGGK" --ew-del "-DSSX" --ew-add "-DSSQ" --ew-add "-DSIG10" --ew-add "-DN1K" --base-ms ${BASE_MS:-40.04} "$@"
