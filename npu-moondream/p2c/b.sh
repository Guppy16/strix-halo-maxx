#!/bin/bash
# usage: b.sh NAME design-args...   (inside npu-iron, /work = npu-moondream)
cd /work/p2c/build && name=$1 && shift
python3 /work/p2c/${DESIGN:-design}.py --out /work/p2c/build/$name "$@" > /work/p2c/build/$name.log 2>&1; echo "rc=$?" >> /work/p2c/build/$name.log
chmod -R a+rwX /work/p2c/build
