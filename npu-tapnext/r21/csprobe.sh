#!/bin/bash
# round 21: csplit (+ --sub) + nresfix + aiecc on a build's lowered MLIR into build/r21/<out> (compile probe; no dispatch)
# usage: r21/csprobe.sh BUILD OUT "csplit args"
cd ~/git/app/.scratch/npu-tapnext
d=build/r21/$2; rm -rf $d; mkdir -p $d
docker run --rm --network none -v $HOME/git/app/.scratch:/work -v $HOME/git/app/strix-halo-maxx:$HOME/git/app/strix-halo-maxx npu-iron bash -c "
  cd /work/npu-tapnext && nice -n 10 python3 r19/csplit.py build/$1.prj/input_with_addresses.mlir $d/in.mlir $3 && python3 r20/nresfix.py $d/in.mlir $d/in.mlir && cd $d &&
  nice -n 10 /opt/ironenv/bin/aiecc in.mlir --unified --peano=/opt/ironenv/lib/python3.13/site-packages/llvm-aie --get-npu-insts \
    --npu-insts-name=\$PWD/cs.bin --get-xclbin --xclbin-name=\$PWD/cs.xclbin --tmpdir=\$PWD/prj --output-dir=\$PWD/prj \
    --get-input-with-addresses --dynamic-objFifos --reclaim-runtime-bds --no-progress > aiecc.log 2>&1; echo aiecc rc=\$?; chmod -R a+rwX /work/npu-tapnext/$d"
grep -i " error" $d/aiecc.log | head -3
