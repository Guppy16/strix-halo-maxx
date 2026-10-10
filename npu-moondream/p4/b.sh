#!/bin/bash
# usage: b.sh NAME design-args...   (inside npu-iron, /work = npu-moondream)
cd /work/p4/build && name=$1 && shift
python3 /work/p4/${DESIGN:-design}.py --out /work/p4/build/$name "$@" > /work/p4/build/$name.log 2>&1; echo "rc=$?" >> /work/p4/build/$name.log
chmod -R a+rwX /work/p4/build
