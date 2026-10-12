#!/bin/bash
# Regression test 4: the SAME code inside the ROCm 7.14 container and on the host (venv-rocm714).
# Runs bench.py (accuracy + median latency on the full test set) and test_throughput.py (nq=1 and nq=2) in both places.
# usage: bash run_docker_vs_host.sh [extra bench.py args]   -> logs_dvh_host.txt / logs_dvh_docker.txt
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)/setup.sh"
cd "$(dirname "$0")"
ARGS="$*"
# MD_FRAMES (private frames) must be set
echo "gpu_busy before: $(cat /sys/class/drm/card1/device/gpu_busy_percent)%"
echo "=== HOST (venv-rocm714: $(venv-rocm714/bin/python -c 'import torch;print(torch.__version__, torch.version.hip)' 2>/dev/null | tail -1)) ==="
( venv-rocm714/bin/python bench.py $ARGS; venv-rocm714/bin/python test_throughput.py --nq 1; venv-rocm714/bin/python test_throughput.py --nq 2 ) 2>&1 | grep -v -E "rocSHMEM|Warn" | tee logs_dvh_host.txt | grep -E "FAST|acc=|sanity|run |config"
echo "=== DOCKER (rocm/dev-ubuntu-24.04:7.14.1-full, venv-c714) ==="
docker run --rm --device /dev/kfd --device /dev/dri --group-add $(getent group render | cut -d: -f3) --security-opt seccomp=unconfined \
  -v "$SHM_SCRATCH:/work" -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -v "$HOME/.local/share/uv:/root/.local/share/uv" -v "$MD_FRAMES:/frames/sessions:ro" \
  -e MD_FRAMES=/frames/sessions -e HF_HUB_OFFLINE=1 -w /work/moondream-opt rocm/dev-ubuntu-24.04:7.14.1-full \
  bash -c "venv-c714/bin/python -c 'import torch;print(torch.__version__, torch.version.hip)'; venv-c714/bin/python bench.py $ARGS; venv-c714/bin/python test_throughput.py --nq 1; venv-c714/bin/python test_throughput.py --nq 2" 2>&1 | grep -v -E "rocSHMEM|Warn" | tee logs_dvh_docker.txt | grep -E "FAST|acc=|sanity|run |config|torch"
echo "gpu_busy after: $(cat /sys/class/drm/card1/device/gpu_busy_percent)%"
