"""In-situ kernel profile of the fast path (eager, torch.profiler): GPU time per kernel for one encode + one query.
usage: python prof_cmp.py [vlin tlin]   e.g. prof_cmp.py HLin HI8Lin"""
import sys, os, torch, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_model, prep, question, SESS
import md2fast, tgemm, hgemm
from md2fast import FastMD2
vl, tl = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("TLin", "I8Lin")
def cls(n):
    for mod in (md2fast, tgemm, hgemm):
        if hasattr(mod, n): return getattr(mod, n)
model = load_model()
fast = FastMD2(model, lin_cls=cls(vl), tlin_cls=cls(tl))
im = prep(f"{SESS}/sessA/judged_0060.jpg")
for _ in range(3):
    fast.set_image(im); fast.set_question(question("mug")); fast.encode(); fast.query()
torch.cuda.synchronize()
from torch.profiler import profile, ProfilerActivity
res = {}
for name, fn in (("encode", fast.encode), ("query", fast.query)):
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(5): fn()
        torch.cuda.synchronize()
    agg = collections.defaultdict(lambda: [0, 0.0])
    tot = 0.0
    for e in p.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            agg[e.name][0] += 1; agg[e.name][1] += e.time_range.elapsed_us() / 5; tot += e.time_range.elapsed_us() / 5
    print(f"== {vl}/{tl} {name}: {tot/1000:.2f} ms GPU kernel time per call ({sum(v[0] for v in agg.values())//5} kernels)")
    for k, (n, t) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:14]:
        print(f"  {t/1000:7.2f} ms  {n//5:4d}x  {k[:110]}")
