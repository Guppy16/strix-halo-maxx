"""Regenerate the public regression clip davis_reg.npy from TAP-Vid DAVIS.

The regress / timing scripts (step8.py regress|time|smoke, common/tools/try.sh) load
$SHM_SCRATCH/tapnet/data/davis_reg.npy (inside containers /work/tapnet/data/davis_reg.npy).
Frames are uint8 (T, H, W, 3); step8.preprocess resizes any HxW to 256x256.

usage: python3 tapnext/npu/make_davis_reg.py [--video goat]
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path(os.environ.get("SHM_SCRATCH", ROOT.parent / ".scratch"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="goat")
    o = ap.parse_args()
    data = SCRATCH / "tapnet" / "data"
    with open(data / "tapvid_davis.pkl", "rb") as f:
        d = pickle.load(f)
    v = np.ascontiguousarray(d[o.video]["video"], dtype=np.uint8)
    assert v.ndim == 4 and v.shape[0] >= 30 and v.shape[-1] == 3, v.shape
    out = data / "davis_reg.npy"
    np.save(out, v)
    print(f"{o.video}: saved {v.shape} {v.dtype} -> {out}")


if __name__ == "__main__":
    main()
