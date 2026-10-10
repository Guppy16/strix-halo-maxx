#!/bin/bash
# usage: b.sh NAME design-args...   (inside npu-iron, /work = npu-moondream)
cd /work/p12/build && name=$1 && shift
python3 /work/p12/${DESIGN:-design}.py --out /work/p12/build/$name "$@" > /work/p12/build/$name.log 2>&1; echo "rc=$?" >> /work/p12/build/$name.log
chmod -R a+rwX /work/p12/build
