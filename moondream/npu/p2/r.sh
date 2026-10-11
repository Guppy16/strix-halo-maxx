#!/bin/bash
# usage: r.sh LOGNAME run.py-args...   (inside npu-iron, needs the NPU device)
export PYTHONPATH=/opt/xilinx/xrt/python:/work/p2
cd /work/p2 && name=$1 && shift
python3 /work/p2/run.py "$@" > /work/p2/build/$name.out 2>&1; echo "rc=$?" >> /work/p2/build/$name.out
