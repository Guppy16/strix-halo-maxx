#!/bin/bash
# usage: b.sh NAME design-args...   (inside npu-iron, /work = npu-moondream)
cd /work/p9/build && name=$1 && shift
python3 /work/p9/${DESIGN:-design}.py --out /work/p9/build/$name "$@" > /work/p9/build/$name.log 2>&1; echo "rc=$?" >> /work/p9/build/$name.log
chmod -R a+rwX /work/p9/build
