#!/bin/bash
# round 21: r20/try20.sh (the round-20 best, full12r20z) + the round-21 kept changes; then csplit20 (CSPLIT + NRESFIX [+ CSUB sub-parts]),
# regress / time / sweep as try20. usage: r21/try21.sh NAME [--sweep] [--ew-add ..] [--k-add ..] [--env K=V]
#   CSUB=0: without the C part-1 sub-parts (round 21: -DCSUB + csplit --sub 1728,3456); ZLOAD=0: without the zero-load first k-step
#   (round 21: ROT_V=g0z from r21/mkzload.py + -DZLOAD); E7=1 [E7_V=e7r68]: research's E7'' kernel (r21/mke7.py), design8 E7 layout, r21/e7patch.py
name=$1; shift
cd ~/git/app/.scratch/npu-tapnext || exit 1
SW=0; for a in "$@"; do [ "$a" = --sweep ] && SW=1; done
R21=()
CSA="--core-rel 1"
if [ "${ZLOAD:-1}" = 1 ]; then R21+=(--env ROT_V=g0z --k-add -DZLOAD); fi
if [ "${E7:-0}" = 1 ]; then R21+=(--env E7=1 --env ROT_V=${E7_V:-e7r68} --k-add -DE7 --k-add -DKS_X0=0x70A00 --k-add -DKS_X1=0x70C40 --k-add -DKS_B1=0x7E180); export KS_ACC_HI=${KS_ACC_HI:-0x71940}; fi
if [ "${CSUB:-1}" = 1 ]; then R21+=(--k-add -DCSUB); CSA="$CSA --sub ${CSUB_AT:-1728,3456}"; fi
NOREG=1 NOTIME=1 NOCS=1 r20/try20.sh "$name" --k-add -DCSPLIT --k-add -DCS_Z=6 --k-add -DEPLR --k-add -DCS_CONS=3 --k-add -DKPREP "${R21[@]}" "$@" || exit 1
E7PATCH=${E7:-0} NRESFIX=1 CSARGS="$CSA" r20/csplit20.sh "$name" | tail -3 || exit 1
[ "${NOREG:-0}" = 1 ] || { r16/nrun.sh ${name}_reg python3 step8.py regress --build full12$name --depth 12 --ref full12r9a | grep "^REGRESS BIT\|^REGRESS DIFF\|^REGRESS frame 3\|rc="; }
[ "${NOTIME:-0}" = 1 ] || { r16/nrun.sh ${name}_time python3 step8.py time --build full12$name --depth 12 --iters 60 --warmup 20 | grep "^TIME\|^PHASES\|rc="; }
[ $SW = 1 ] && { TMO=1500 r16/nrun.sh ${name}_sweep python3 step8.py sweep --build full12$name --iters 60 --warmup 20 | grep "^SWEEP\|^PHASES\|rc="; }
exit 0
