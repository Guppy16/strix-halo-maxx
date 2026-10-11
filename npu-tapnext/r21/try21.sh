#!/bin/bash
# round 21: r20/try20.sh (the round-20 best, full12r20z) + the round-21 kept changes; then csplit20 (CSPLIT + NRESFIX [+ CSUB sub-parts]),
# regress / time / sweep as try20. usage: r21/try21.sh NAME [--sweep] [--ew-add ..] [--k-add ..] [--env K=V]
#   CSUB=0: without the C part-1 sub-parts (round 21: -DCSUB + csplit --sub 1728,3456)
name=$1; shift
cd ~/git/app/.scratch/npu-tapnext || exit 1
SW=0; for a in "$@"; do [ "$a" = --sweep ] && SW=1; done
R21=()
CSA="--core-rel 1"
if [ "${CSUB:-0}" = 1 ]; then R21+=(--k-add -DCSUB); CSA="$CSA --sub ${CSUB_AT:-1728,3456}"; fi
NOREG=1 NOTIME=1 NOCS=1 r20/try20.sh "$name" "${R21[@]}" --k-add -DCSPLIT --k-add -DCS_Z=6 --k-add -DEPLR --k-add -DCS_CONS=3 --k-add -DKPREP "$@" || exit 1
NRESFIX=1 CSARGS="$CSA" r20/csplit20.sh "$name" | tail -3 || exit 1
[ "${NOREG:-0}" = 1 ] || { r16/nrun.sh ${name}_reg python3 step8.py regress --build full12$name --depth 12 --ref full12r9a | grep "^REGRESS BIT\|^REGRESS DIFF\|^REGRESS frame 3\|rc="; }
[ "${NOTIME:-0}" = 1 ] || { r16/nrun.sh ${name}_time python3 step8.py time --build full12$name --depth 12 --iters 60 --warmup 20 | grep "^TIME\|^PHASES\|rc="; }
[ $SW = 1 ] && { TMO=1500 r16/nrun.sh ${name}_sweep python3 step8.py sweep --build full12$name --iters 60 --warmup 20 | grep "^SWEEP\|^PHASES\|rc="; }
exit 0
