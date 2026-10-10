#!/bin/bash
# round 16: npu-research/tools/try.sh on the current best (full12r16a = r15m + SQK). usage: r16/try16.sh NAME [try.sh options]
name=$1; shift
exec ~/git/app/.scratch/npu-research/tools/try.sh "$name" --ew-add "-DSQK" --base-ms ${BASE_MS:-41.32} "$@"
