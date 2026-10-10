#!/usr/bin/env python
"""GPU side of the pipelined frame: moondream's pooled text work only (pooled text prefill from
cached projected image tokens + the question query), CUDA-graphed like bench.py, looped for
--seconds with per-frame timing. The vision encoder runs once to produce the cached tokens.

  venv-rocm714/bin/python textloop.py --seconds 40 --pool 16 [--full]   (--full: whole pooled path incl. vision, upper bound)
"""
import argparse, os, sys, time, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
from common import load_model, prep, question, SESS
import md2fast, tgemm, hgemm
from md2fast import FastMD2, GRID

ap = argparse.ArgumentParser()
ap.add_argument("--pool", type=int, default=16)
ap.add_argument("--seconds", type=float, default=40)
ap.add_argument("--full", action="store_true", help="run the whole pooled path (vision + text) per frame")
ap.add_argument("--label", default="gpu")
args = ap.parse_args()


def cls(n):
    for mod in (md2fast, tgemm, hgemm):
        if hasattr(mod, n):
            return getattr(mod, n)
    raise KeyError(n)


model = load_model()
fast = FastMD2(model, lin_cls=cls("HLin"), tlin_cls=cls("HI8Lin"), pool=args.pool, fuse=True, n_q=1, hattn=False)
fast.set_image(prep(f"{SESS}/sessA/judged_0060.jpg")); fast.set_questions([question("mug")])
for _ in range(3):
    fast.encode(); fast.query()
torch.cuda.synchronize()

if not args.full:
    # cache the pooled projected image tokens from one real vision pass, then replace _encode by the text part only
    with torch.inference_mode():
        emb = fast._vision(fast.img_buf)
        emb = F.adaptive_avg_pool2d(emb.view(GRID, GRID, -1).permute(2, 0, 1), (fast.pool, fast.pool)).permute(1, 2, 0).reshape(fast.n_img, -1)
        fast.emb_cache = emb.clone()

    def _encode_text_only(self):
        x = torch.cat([F.embedding(self.bos, self.wte), self.emb_cache[None]], dim=1)
        self._text_layers(x, 0, self.prefix, self.cos_pre, self.sin_pre, None)

    fast._encode = types.MethodType(_encode_text_only, fast)
fast.capture()
torch.cuda.synchronize()
ref = fast.decide(fast.query())[0]


def gpu_busy():
    try:
        return int(open("/sys/class/drm/card1/device/gpu_busy_percent").read())
    except OSError:
        return -1


t_start = time.time()
enc, qry, busy = [], [], []
n = 0
while time.time() - t_start < args.seconds:
    torch.cuda.synchronize(); t0 = time.perf_counter()
    fast.encode(); torch.cuda.synchronize(); t1 = time.perf_counter()
    lg = fast.query(); torch.cuda.synchronize(); t2 = time.perf_counter()
    enc.append(t1 - t0); qry.append(t2 - t1); n += 1
    if n % 5 == 0:
        busy.append(gpu_busy())
t_end = time.time()
e, q = 1e3 * np.array(enc), 1e3 * np.array(qry)
tot = e + q
print(
    f"GPU[{args.label}] {t_start:.0f}-{t_end:.0f} frames {n} {'full-pooled' if args.full else 'text-only'} frame_ms median {np.median(tot):.1f}"
    f" p10 {np.percentile(tot, 10):.1f} p90 {np.percentile(tot, 90):.1f} min {tot.min():.1f} (encode {np.median(e):.1f}, query {np.median(q):.1f})"
    f" | gpu_busy median {np.median(busy) if busy else -1:.0f}% | answer {'yes' if fast.decide(lg)[0] else 'no'} (warmup {'yes' if ref else 'no'})",
    flush=True,
)
