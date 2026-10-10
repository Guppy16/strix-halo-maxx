#!/bin/bash
# round 18: r17/xtry.sh (XROT=4 phase builds + regress + time) on the current round-18 best (r17d + HRING PTRG ZK KSC) plus your
# changes. usage: r18/try18.sh NAME [--sweep] [--ew-add ..] [--k-add ..] [--env K=V]   (NOREG=1 / NOTIME=1 skip those steps)
# KSC: the addresses are checked by design8 after the link (ksc_check); a layout change fails the build and names the new ones.
name=$1; shift
cd ~/git/app/.scratch/npu-tapnext || exit 1
KS="--k-add -DKSC --k-add -DKS_ACC_HI=${KS_ACC_HI:-0x71900} --k-add -DKS_C=0x78900 --k-add -DKS_A0=0x77B80 --k-add -DKS_A1=0x78000 --k-add -DKS_A2=0x78480"
BASE_MS=${BASE_MS:-38.72} exec r17/xtry.sh "$name" 4 --k-add -DATTKS2MAX --env AL1=1 --ew-add -DAL1 --ew-add -DNPIPE --ew-add -DNPIPE_G=6 \
  --ew-add -DHRING --ew-add -DPTRG --ew-add -DZK --ew-add -DNIOC --ew-add -DCGF --k-add -DV2T $KS "$@"
