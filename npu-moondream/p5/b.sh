#!/bin/bash
# usage: b.sh NAME design-args...   (inside npu-iron, /work = npu-moondream)
cd /work/p5/build && name=$1 && shift
python3 /work/p5/${DESIGN:-design}.py --out /work/p5/build/$name "$@" > /work/p5/build/$name.log 2>&1; echo "rc=$?" >> /work/p5/build/$name.log
chmod -R a+rwX /work/p5/build
