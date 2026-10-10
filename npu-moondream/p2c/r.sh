#!/bin/bash
# usage: r.sh LOGNAME run.py-args...   (inside npu-iron, needs the NPU device)
export PYTHONPATH=/opt/xilinx/xrt/python:/work/p2c
cd /work/p2c && name=$1 && shift
python3 /work/p2c/run.py "$@" > /work/p2c/build/$name.out 2>&1; echo "rc=$?" >> /work/p2c/build/$name.out
