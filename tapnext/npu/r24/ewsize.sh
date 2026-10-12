#!/bin/bash
SHM_REPO=$(cd "$(dirname "$0")" && git rev-parse --show-toplevel); SHM_SCRATCH=${SHM_SCRATCH:-$SHM_REPO/../.scratch}
# round 24 (copy of npu-research/r23/ewsize.sh with TXT18, output in r24/): compile an ew8.cc variant with the deployed EW flags (full12r22c: EW_OPT -O2 + EW_DEFS) and print its C++ text and
# the largest functions. CPU only (npu-iron). usage (from $SHM_SCRATCH): npu-research/r23/ewsize.sh [SRC] [extra -D ...]
#   SRC defaults to npu-tapnext/ew8.cc; a variant must sit in npu-tapnext/ (it includes "build/*.h" relative to itself)
SRC=${1:-npu-tapnext/ew8.cc}; shift
EWD="-DSSM_LUT -DSSM_EXP3 -DSIGK -DCPY64 -DCONVK -DRECK -DNRMK -DZRING -DSQK -DINPART_INLINE -DXONE -DGGK -DSSQ -DSIG10 -DN1K -DXROT -DAL1 -DNPIPE -DNPIPE_G=6 -DHRING -DPTRG -DZK -DNIOC -DCGF -DNRES -DTXT18"
FL="-I/work/npu-moondream/mlir-aie/aie_kernels -DEW_IN_SLOTS=20 -DEW_OUT_SLOTS=12 -DL_IN_FREE=0 -DL_IN_FILL=1 -DL_OUT_FREE=2 -DL_OUT_FILL=3 -DL_P_FREE=4 -DL_P_FILL=5 -DPATCH_PARAM_B=14592 -DSSM_PARAM_B=12288 -O2 $EWD $*"
P=/opt/ironenv/lib/python3.13/site-packages/llvm-aie
O=npu-tapnext/r24/.ew
docker run --rm --network none -v $SHM_SCRATCH:/work -v $SHM_REPO:$SHM_REPO npu-iron bash -c "
  cd /work && $P/bin/clang++ -std=c++20 -D__AIE_API_AIE_ADF_HPP__ -Wno-attributes -w --target=aie2p-none-unknown-elf \
   -I$P/include/aie2p-none-unknown-elf -I/opt/ironenv/lib/python3.13/site-packages/mlir_aie/include -I/work/npu-tapnext $FL \
   -c $SRC -o $O.o && $P/bin/llvm-size $O.o | tail -1 > $O.size && $P/bin/llvm-nm --size-sort -S --radix=d $O.o > $O.nm && \
   ${DIS:+$P/bin/llvm-objdump -d -z --no-show-raw-insn $O.o > $DIS &&} chmod a+rw $O.* ${DIS:-}"
cat $O.size
grep -i ' t ' $O.nm | awk '{printf "%6d %s\n", $2 + 0, $4}' | tail -${NTOP:-14}
rm -f $O.o $O.size $O.nm
