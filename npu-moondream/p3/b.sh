#!/bin/bash
# usage: b.sh NAME design-args...   (inside npu-iron, /work = npu-moondream)
cd /work/p3/build && name=$1 && shift
python3 /work/p3/${DESIGN:-design}.py --out /work/p3/build/$name "$@" > /work/p3/build/$name.log 2>&1; echo "rc=$?" >> /work/p3/build/$name.log
chmod -R a+rwX /work/p3/build
