#!/bin/bash
# usage: r.sh LOGNAME run.py-args...   (inside npu-iron, needs the NPU device)
export PYTHONPATH=/opt/xilinx/xrt/python:/work/p8
cd /work/p8 && name=$1 && shift
python3 /work/p8/run.py "$@" > /work/p8/build/$name.out 2>&1; echo "rc=$?" >> /work/p8/build/$name.out
