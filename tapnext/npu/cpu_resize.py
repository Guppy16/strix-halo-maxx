"""CPU cost of the frame preprocessing (resize to 256x256 bilinear + normalise to [-1, 1] bf16), per thread count.
    python3 cpu_resize.py
"""

import time

import numpy as np
import torch
import torch.nn.functional as F


def prep(frame_u8, dtype):
    t = torch.from_numpy(frame_u8).permute(2, 0, 1)[None].float()
    t = F.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False, antialias=False)
    return (t[0] / 127.5 - 1.0).permute(1, 2, 0).contiguous().to(dtype)


def prep_aa(frame_u8, dtype):
    t = torch.from_numpy(frame_u8).permute(2, 0, 1)[None].float()
    t = F.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False, antialias=True)
    return (t[0] / 127.5 - 1.0).permute(1, 2, 0).contiguous().to(dtype)


def main():
    for shape in ((256, 256, 3), (720, 1280, 3), (1080, 1920, 3)):
        frame = np.random.randint(0, 255, shape, dtype=np.uint8)
        for threads in (1, 4):
            torch.set_num_threads(threads)
            for fn, label in ((prep, "bilinear"), (prep_aa, "bilinear+antialias")):
                for _ in range(10):
                    fn(frame, torch.bfloat16)
                ts = []
                for _ in range(100):
                    t0 = time.perf_counter()
                    fn(frame, torch.bfloat16)
                    ts.append(time.perf_counter() - t0)
                a = np.array(ts) * 1e3
                print(f"{shape[1]}x{shape[0]} -> 256x256 {label:20s} threads {threads}: median {np.median(a):.2f} ms p99 {np.percentile(a, 99):.2f}")


if __name__ == "__main__":
    main()
