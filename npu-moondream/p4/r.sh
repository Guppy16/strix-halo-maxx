#!/bin/bash
# usage: r.sh LOGNAME run.py-args...   (inside npu-iron, needs the NPU device)
export PYTHONPATH=/opt/xilinx/xrt/python:/work/p4
cd /work/p4 && name=$1 && shift
python3 /work/p4/run.py "$@" > /work/p4/build/$name.out 2>&1; echo "rc=$?" >> /work/p4/build/$name.out
