#!/bin/bash
# usage: b.sh NAME design-args...   (inside npu-iron, /work = npu-moondream)
cd /work/p11/build && name=$1 && shift
python3 /work/p11/${DESIGN:-design}.py --out /work/p11/build/$name "$@" > /work/p11/build/$name.log 2>&1; echo "rc=$?" >> /work/p11/build/$name.log
chmod -R a+rwX /work/p11/build
