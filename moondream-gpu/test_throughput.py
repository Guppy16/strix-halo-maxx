"""Regression test 3 (throughput): frames/second of the fast path in a tight loop over >= 200 distinct test frames,
with 1 and 2 questions per frame.  Images are decoded + squashed up front (CPU prep is reported separately), the loop does
H2D copy, encode, query, decision.  Same CLI options as bench.py for the GEMM implementations.
  python test_throughput.py [--vlin HLin --tlin HI8Lin] [--frames 200] [--nq 2]"""
import argparse, os, sys, time, statistics, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_model, load_testset, prep, question, SESS
import md2fast, tgemm, hgemm
from md2fast import FastMD2
ap = argparse.ArgumentParser()
ap.add_argument("--vlin", default="HLin"); ap.add_argument("--tlin", default="HI8Lin")
ap.add_argument("--frames", type=int, default=200); ap.add_argument("--nq", type=int, default=1); ap.add_argument("--pool", type=int, default=0)
ap.add_argument("--repeat", type=int, default=3, help="how many times to run the loop (report each)")
args = ap.parse_args()
def cls(n):
    for mod in (md2fast, tgemm, hgemm):
        if hasattr(mod, n): return getattr(mod, n)
    raise KeyError(n)
model = load_model()
fast = FastMD2(model, lin_cls=cls(args.vlin), tlin_cls=cls(args.tlin), pool=args.pool, fuse=True, n_q=args.nq)
objs = ["mug", "tv"]
items = load_testset(args.frames)
t0 = time.time(); imgs = [prep(it["path"]) for it in items]; tprep = (time.time() - t0) / len(items) * 1000
import numpy as np
tens = [torch.from_numpy(np.array(im.convert("RGB"))) for im in imgs]
qs = [[question(objs[(i + j) % 2]) for j in range(args.nq)] for i in range(len(items))]
for _ in range(3):
    fast.set_image(tens[0]); fast.set_questions(qs[0]); fast.encode(); fast.query()
fast.capture(); torch.cuda.synchronize()
busy = open("/sys/class/drm/card1/device/gpu_busy_percent").read().strip() if os.path.exists("/sys/class/drm/card1/device/gpu_busy_percent") else "?"
print(f"config vlin={args.vlin} tlin={args.tlin} nq={args.nq} pool={args.pool} frames={len(items)} | CPU prep (decode+LANCZOS) {tprep:.1f} ms/frame (not in the loop) | gpu_busy before={busy}%")
for r in range(args.repeat):
    torch.cuda.synchronize(); t0 = time.perf_counter(); nyes = 0
    for i in range(len(items)):
        fast.set_image(tens[i]); fast.set_questions(qs[i]); fast.encode(); lg = fast.query()
        nyes += sum(int(p) for p, _, _ in fast.decide_all(lg))
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    print(f"run {r}: {len(items)} frames x {args.nq} question(s) in {dt*1000:.0f} ms -> {len(items)/dt:.2f} frames/s, {len(items)*args.nq/dt:.2f} answers/s ({dt/len(items)*1000:.1f} ms/frame), yes={nyes}", flush=True)
