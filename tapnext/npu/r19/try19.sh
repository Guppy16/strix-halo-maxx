#!/bin/bash
SHM_REPO=$(cd "$(dirname "$0")" && git rev-parse --show-toplevel); SHM_SCRATCH=${SHM_SCRATCH:-$SHM_REPO/../.scratch}
# round 19: r18/try18.sh on the round-19 best (r18m + ROT g0a + ZHI + CSPLIT + EPLR), then r19/csplit.sh (the C drain split is patched
# into the lowered MLIR and the xclbin rebuilt BEFORE anything dispatches: an unpatched -DCSPLIT xclbin hangs), then regress / time /
# sweep. usage: r19/try19.sh NAME [--sweep] [--ew-add ..] [--k-add ..] [--env K=V]
#   NOREG=1 / NOTIME=1 skip those; NOCS=1: no CSPLIT / EPLR; KS_ACC_HI=0x.. when design8's ksc_check names a new acc_hi
name=$1; shift
cd "$SHM_SCRATCH/npu-tapnext" || exit 1
SW=0; for a in "$@"; do [ "$a" = --sweep ] && SW=1; done
CS=(--k-add -DCSPLIT --k-add -DCS_Z=6 --k-add -DEPLR --k-add -DCS_CONS=3 --k-add -DKPREP); [ "${NOCS:-0}" = 1 ] && CS=()
KS_ACC_HI=${KS_ACC_HI:-0x719C0} NOREG=1 NOTIME=1 NOSWEEP=1 r18/try18.sh "$name" --env ROT_V=g0a --env SEQPRE=${SEQPRE:-1} --k-add -DZHI "${CS[@]}" "$@"
tail -1 build/full12$name.log | grep -q rc=0 || { echo "BUILD FAILED"; exit 1; }
if [ "${NOCS:-0}" != 1 ]; then CSARGS="--core-rel 1" r19/csplit.sh "$name" | tail -2 || exit 1; fi
[ "${NOREG:-0}" = 1 ] || { r16/nrun.sh ${name}_reg python3 step8.py regress --build full12$name --depth 12 --ref full12r9a | grep "^REGRESS BIT\|^REGRESS DIFF\|^REGRESS frame 3\|rc="; }
[ "${NOTIME:-0}" = 1 ] || { r16/nrun.sh ${name}_time python3 step8.py time --build full12$name --depth 12 --iters 60 --warmup 20 | grep "^TIME\|^PHASES\|rc="; }
[ $SW = 1 ] && { TMO=1500 r16/nrun.sh ${name}_sweep python3 step8.py sweep --build full12$name --iters 60 --warmup 20 | grep "^SWEEP\|^PHASES\|rc="; }
exit 0
