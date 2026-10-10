#!/bin/bash
# usage: run_shapes.sh [extra harness args]   -- runs inside npu-iron, results to stdout
E=/work/mlir-aie/programming_examples/ml/block_datatypes/gemm_asymmetric_tile_buffering
cd /work/atb; mkdir -p build
sed -i '1s/^/#pragma GCC optimize("O3")\n/' test.cpp 2>/dev/null; grep -q 'pragma GCC optimize' test.cpp || true
mkdir -p host && cd host
[ -x n32_core.exe ] || make -f $E/config1/Makefile hostsrc=/work/atb/test.cpp n32_core.exe > build.log 2>&1 || { tail -20 build.log; exit 1; }
cd /work/atb; mkdir -p build
# name Mr Kr Nr  M K N  m n mtk
while read name Mr Kr Nr M K N m n mtk; do
  [ -z "$name" ] && continue
  tag=${M}x${K}x${N}_${m}x${n}_${mtk}
  if [ ! -f build/$tag.xclbin ]; then
    python3 enc_gemm.py --dev npu2 -M $M -K $K -N $N -m $m -k 64 -n $n --mtk $mtk \
      --xclbin-path build/$tag.xclbin --insts-path build/$tag.bin > build/$tag.log 2>&1 \
      || { echo "$name $tag COMPILE FAILED"; tail -5 build/$tag.log; continue; }
  fi
  out=$(timeout 120 host/n32_core.exe -x build/$tag.xclbin -i build/$tag.bin -k MLIR_AIE -M $M -K $K -N $N -w $m -y 64 -z $n \
        --Mr $Mr --Kr $Kr --Nr $Nr --warmup 10 --iters 30 -v 0 "$@" 2>&1)
  rc=$?
  echo "$name ${Mr}x${Kr}x${Nr} pad $tag rc=$rc | $(echo "$out" | grep -E '^MED') | $(echo "$out" | grep -E '^ERR')"
  [ $rc -ne 0 ] && echo "$out" | tail -5
done < ${SHAPES:-shapes.txt}
