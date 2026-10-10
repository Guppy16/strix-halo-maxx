#!/bin/bash
# usage: b.sh NAME design-args...   (inside npu-iron, /work = npu-moondream)
cd /work/p7/build && name=$1 && shift
python3 /work/p7/${DESIGN:-design}.py --out /work/p7/build/$name "$@" > /work/p7/build/$name.log 2>&1; echo "rc=$?" >> /work/p7/build/$name.log
chmod -R a+rwX /work/p7/build
