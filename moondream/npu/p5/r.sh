#!/bin/bash
# usage: r.sh LOGNAME run.py-args...   (inside npu-iron, needs the NPU device)
export PYTHONPATH=/opt/xilinx/xrt/python:/work/p5
cd /work/p5 && name=$1 && shift
python3 /work/p5/run.py "$@" > /work/p5/build/$name.out 2>&1; echo "rc=$?" >> /work/p5/build/$name.out
