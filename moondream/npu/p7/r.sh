#!/bin/bash
# usage: r.sh LOGNAME run.py-args...   (inside npu-iron, needs the NPU device)
export PYTHONPATH=/opt/xilinx/xrt/python:/work/p7
cd /work/p7 && name=$1 && shift
python3 /work/p7/run.py "$@" > /work/p7/build/$name.out 2>&1; echo "rc=$?" >> /work/p7/build/$name.out
