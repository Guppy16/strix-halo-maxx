#!/bin/bash
# round 18: r17/xtry.sh (XROT=4 phase builds + regress + time) on the round-17 best (full12r17d) plus your changes.
# usage: r18/try18.sh NAME [--sweep] [--ew-add ..] [--k-add ..] [--env K=V]   (NOREG=1 / NOTIME=1 skip those steps)
name=$1; shift
cd ~/git/app/.scratch/npu-tapnext || exit 1
BASE_MS=${BASE_MS:-39.16} exec r17/xtry.sh "$name" 4 --k-add -DATTKS2MAX --env AL1=1 --ew-add -DAL1 --ew-add -DNPIPE --ew-add -DNPIPE_G=6 "$@"
