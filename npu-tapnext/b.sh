#!/bin/bash
# usage: b.sh NAME design-args...   (inside npu-iron, /work = npu-moondream)
cd /work/npu-tapnext/build && name=$1 && shift
python3 /work/npu-tapnext/${DESIGN:-design}.py --out /work/npu-tapnext/build/$name "$@" > /work/npu-tapnext/build/$name.log 2>&1; echo "rc=$?" >> /work/npu-tapnext/build/$name.log
chmod -R a+rwX /work/npu-tapnext/build
