"""Round 12: decode r12/gen_pushprobe.py dumps (r11/apb.py --dump) against pack7.encode_hw.

    python3 r12/pushcheck.py build/r12/apbP      (reads apbP_p{k}_l0_n144_i0.npy for every probe present)
"""

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pack7 import encode_hw  # noqa: E402


def inputs():
    i = np.arange(4608)
    return ((i * 37) % 2048 - 1024).astype(np.float32) / np.float32(200.0)


def blocks(x32, n):
    return [encode_hw(x32[64 * b:64 * b + 64].reshape(1, 64)).ravel().tobytes() for b in range(n)]


def find(out, want):
    """offsets of each wanted 72 B block in the dump (-1 = absent)"""
    return [out.find(w) for w in want]


def main():
    base = sys.argv[1]
    x = inputs()
    cands = {"rne": x.astype(bfloat16).astype(np.float32)}
    for k in range(8):
        f = Path(f"{base}_p{k}_l0_n144_i0.npy")
        if not f.exists():
            continue
        out = np.load(f).astype(np.uint16).tobytes()
        sent = np.frombuffer(out, np.uint16) == 0x7FC1
        written = np.flatnonzero(~np.repeat(sent, 2))
        spans = []
        if len(written):
            s = written[0]
            for a, b in zip(written, written[1:]):
                if b != a + 1:
                    spans.append((s, a + 1))
                    s = b
            spans.append((s, written[-1] + 1))
        for nm, xv in cands.items():
            off = find(out, blocks(xv, 12))
            print(f"probe {k} [{nm}] block offsets {off}  written byte spans {spans}")


if __name__ == "__main__":
    main()
