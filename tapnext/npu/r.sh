#!/bin/bash
# usage: r.sh LOGNAME run.py-args...   (inside npu-iron, /work = .scratch, needs the NPU device)
export PYTHONPATH=/opt/xilinx/xrt/python:/work/npu-tapnext
cd /work/npu-tapnext && name=$1 && shift
python3 /work/npu-tapnext/run.py "$@" > /work/npu-tapnext/build/$name.out 2>&1; echo "rc=$?" >> /work/npu-tapnext/build/$name.out
