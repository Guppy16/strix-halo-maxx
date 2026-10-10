"""Robust A/B of the encode graph with torch SDPA vs the HIP attention kernel: both captured in one process, replays alternated, min + median."""
import sys, os, time, statistics, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_model, prep, question, SESS
import hgemm
from md2fast import FastMD2
model = load_model()
im = prep(f"{SESS}/sessA/judged_0060.jpg")
fasts = {}
for name, ha in (("sdpa", False), ("hattn", True)):
    f = FastMD2(model, lin_cls=hgemm.HLin, tlin_cls=hgemm.HI8Lin, hattn=ha)
    for _ in range(3): f.set_image(im); f.set_question(question("mug")); f.encode(); f.query()
    f.capture(); fasts[name] = f
torch.cuda.synchronize()
res = {k: [] for k in fasts}
for it in range(40):
    for name, f in fasts.items():
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); f.encode(); e1.record(); torch.cuda.synchronize(); res[name].append(e0.elapsed_time(e1))
busy = open("/sys/class/drm/card1/device/gpu_busy_percent").read().strip()
for name, ts in res.items():
    print(f"encode graph {name:6s}: min {min(ts):.2f} ms  median {statistics.median(ts):.2f} ms  (gpu_busy={busy}%)")
lg0 = fasts["sdpa"].query(); lg1 = fasts["hattn"].query(); torch.cuda.synchronize()
print("yes/no logits sdpa vs hattn:", lg0.float().tolist(), lg1.float().tolist())
