#!/bin/bash
# usage: r.sh LOGNAME run.py-args...   (inside npu-iron, needs the NPU device)
export PYTHONPATH=/opt/xilinx/xrt/python:/work/p6
cd /work/p6 && name=$1 && shift
python3 /work/p6/run.py "$@" > /work/p6/build/$name.out 2>&1; echo "rc=$?" >> /work/p6/build/$name.out
