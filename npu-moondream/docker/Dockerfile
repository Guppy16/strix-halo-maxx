# npu-iron: XRT 2.21 userspace (the maintainer's own build, Ubuntu 25.10 debs, matches host amdxdna 2.21)
# + IRON/mlir-aie v1.4.4 wheels + Peano (llvm-aie). Python 3.13 (pyxrt in the debs is cp313).
FROM ubuntu:25.10
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential clang lld cmake ninja-build git curl ca-certificates jq \
      python3 python3-venv python3-pip python3-dev uuid-dev pciutils \
    && rm -rf /var/lib/apt/lists/*
COPY debs /tmp/debs
# base/base-dev/npu via apt (pulls boost 1.88, protobuf, ocl-icd). The plugin deb's
# postinst runs dkms, so only extract its userspace shim (libxrt_driver_xdna).
RUN apt-get update && apt-get install -y --no-install-recommends \
      /tmp/debs/xrt_202610.2.21.0_25.10-amd64-base.deb \
      /tmp/debs/xrt_202610.2.21.0_25.10-amd64-base-dev.deb \
      /tmp/debs/xrt_202610.2.21.0_25.10-amd64-npu.deb \
    && dpkg-deb -x /tmp/debs/xrt_plugin.2.21.0_25.10-amd64-amdxdna.deb /tmp/plugin \
    && cp -a /tmp/plugin/opt/xilinx/xrt/lib/libxrt_driver_xdna.so* /opt/xilinx/xrt/lib/ \
    && cp -a /tmp/plugin/opt/xilinx/xrt/share/amdxdna /opt/xilinx/xrt/share/ \
    && rm -rf /tmp/debs /tmp/plugin /var/lib/apt/lists/*
ENV XILINX_XRT=/opt/xilinx/xrt \
    PATH=/opt/xilinx/xrt/bin:$PATH \
    LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
    PYTHONPATH=/opt/xilinx/xrt/python
RUN python3 -m venv /opt/ironenv
ENV PATH=/opt/ironenv/bin:$PATH
COPY peano-requirements.txt /tmp/peano-requirements.txt
RUN pip install --upgrade pip \
    && pip install "mlir_aie==1.4.4" -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.4.4 \
    && pip install -r /tmp/peano-requirements.txt \
    && pip install numpy ml_dtypes \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN SP=$(python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])') \
    && echo "$SP/mlir_aie/python" > "$SP/mlir-aie.pth"
ENV MLIR_AIE_INSTALL_DIR=/opt/ironenv/lib/python3.13/site-packages/mlir_aie \
    PEANO_INSTALL_DIR=/opt/ironenv/lib/python3.13/site-packages/llvm-aie
ENV PATH=/opt/ironenv/lib/python3.13/site-packages/mlir_aie/bin:$PATH
WORKDIR /work
