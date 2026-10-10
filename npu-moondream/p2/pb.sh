#!/bin/bash
# usage: pb.sh "name args" ...  build+run bw probes (inside npu-iron with the device)
export PYTHONPATH=/opt/xilinx/xrt/python:/work/p2
cd /work/p2/build
for spec in "$@"; do set -- $spec; name=$1; shift
  [ -f $name.xclbin ] || python3 /work/p2/bw.py --out $name "$@" > $name.log 2>&1 || { echo "$name BUILD FAIL"; grep -m2 -i error $name.log; continue; }
  timeout 60 python3 /work/p2/runbw.py $name 2>&1 | tail -1
done
chmod -R a+rwX /work/p2/build
