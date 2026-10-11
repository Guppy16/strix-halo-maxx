#!/bin/bash
# usage: r.sh LOGNAME run.py-args...   (inside npu-iron, needs the NPU device)
export PYTHONPATH=/opt/xilinx/xrt/python:/work/p3
cd /work/p3 && name=$1 && shift
python3 /work/p3/run.py "$@" > /work/p3/build/$name.out 2>&1; echo "rc=$?" >> /work/p3/build/$name.out
