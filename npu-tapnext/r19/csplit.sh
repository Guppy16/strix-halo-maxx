#!/bin/bash
# round 19 CSPLIT: patch build/full12NAME's lowered MLIR (r19/csplit.py) and rebuild its xclbin with aiecc from it (the runtime
# sequence .bin must come out byte-identical: the patch touches only core-tile DMAs, locks and core code). Run RIGHT AFTER the build
# (aiecc relinks the kernel objects build/*.o, which the next design build overwrites). The unpatched xclbin of a -DCSPLIT build
# hangs (lock CS_Z does not exist): it is renamed to .unpatched.xclbin.
# usage: r19/csplit.sh NAME
set -e
n=full12$1
cd ~/git/app/.scratch/npu-tapnext
d=build/r19/cs_$1
rm -rf $d; mkdir -p $d
docker run --rm --network none -v $HOME/git/app/.scratch:/work -v $HOME/git/app/strix-halo-maxx:$HOME/git/app/strix-halo-maxx npu-iron bash -c "
  cd /work/npu-tapnext && nice -n 10 python3 r19/csplit.py build/$n.prj/input_with_addresses.mlir $d/in.mlir ${CSARGS:-} && cd $d &&
  nice -n 10 /opt/ironenv/bin/aiecc in.mlir --unified --peano=/opt/ironenv/lib/python3.13/site-packages/llvm-aie --get-npu-insts \
    --npu-insts-name=\$PWD/cs.bin --get-xclbin --xclbin-name=\$PWD/cs.xclbin --tmpdir=\$PWD/prj --output-dir=\$PWD/prj \
    --get-input-with-addresses --dynamic-objFifos --reclaim-runtime-bds --no-progress > aiecc.log 2>&1; echo aiecc rc=\$?;
  chmod -R a+rwX /work/npu-tapnext/$d"
cmp $d/cs.bin build/$n.bin && echo "csplit: runtime sequence unchanged"
grep -c "csz_" $d/prj/input_with_addresses.mlir
[ -e build/$n.unpatched.xclbin ] || mv build/$n.xclbin build/$n.unpatched.xclbin
cp $d/cs.xclbin build/$n.xclbin
echo "csplit: build/$n.xclbin patched"
