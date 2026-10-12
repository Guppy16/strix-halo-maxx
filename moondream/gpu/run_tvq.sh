#!/bin/bash
# The TV-question wordings, each through the GTT-guarded runner (exact path, 729 tokens).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)/setup.sh"
cd "$SHM_SCRATCH/moondream-opt"
py=venv-rocm714/bin/python
./gpu_safe.sh runs-1007/tvq_base.log $py frame_preds.py --name q_base
./gpu_safe.sh runs-1007/tvq_tele.log $py frame_preds.py --name q_television \
    --tv-question "Is there a television in this image? Answer yes or no."
./gpu_safe.sh runs-1007/tvq_nomon.log $py frame_preds.py --name q_no_monitor \
    --tv-question "Is there a TV in this image? A computer monitor does not count. Answer yes or no."
./gpu_safe.sh runs-1007/tvq_set.log $py frame_preds.py --name q_tv_set \
    --tv-question "Is there a television set (not a computer monitor) in this image? Answer yes or no."
tail -qn1 runs-1007/tvq_*.log
