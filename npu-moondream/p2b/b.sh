#!/bin/bash
# usage: b.sh NAME design-args...   (inside npu-iron, /work = npu-moondream)
cd /work/p2b/build && name=$1 && shift
python3 /work/p2b/${DESIGN:-design}.py --out /work/p2b/build/$name "$@" > /work/p2b/build/$name.log 2>&1; echo "rc=$?" >> /work/p2b/build/$name.log
chmod -R a+rwX /work/p2b/build
