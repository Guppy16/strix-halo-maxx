"""In-situ per-GEMM timing (eager, CUDA events around every linear call) for HIP vs Triton implementations."""
import sys, os, torch, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_model, prep, question, SESS
import md2fast, tgemm, hgemm
from md2fast import FastMD2
model = load_model()
im = prep(f"{SESS}/sessA/judged_0060.jpg")
def wrap(cls, tag, rec):
    orig = cls.__call__
    def call(self, x, *a, **k):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); out = orig(self, x, *a, **k); e1.record()
        rec.append((tag, x.reshape(-1, self.K).shape[0], self.N, self.K, e0, e1))
        return out
    cls.__call__ = call
for vl, tl in (("TLin", "I8Lin"), ("HLin", "HI8Lin")):
    rec = []
    V = getattr(hgemm, vl, None) or getattr(tgemm, vl); T = getattr(hgemm, tl, None) or getattr(tgemm, tl)
    wrap(V, vl, rec); wrap(T, tl, rec)
    fast = FastMD2(model, lin_cls=V, tlin_cls=T)
    for _ in range(3):
        fast.set_image(im); fast.set_question(question("mug")); fast.encode(); fast.query()
    torch.cuda.synchronize(); rec.clear()
    R = 5
    for _ in range(R):
        fast.encode(); fast.query()
    torch.cuda.synchronize()
    agg = collections.defaultdict(lambda: [0, 0.0])
    for tag, M, N, K, e0, e1 in rec:
        a = agg[(tag, M, N, K)]; a[0] += 1; a[1] += e0.elapsed_time(e1)
    print(f"== {vl}/{tl}: per-call in-situ (eager) GEMM times, {R} encode+query")
    tot = 0
    for (tag, M, N, K), (n, t) in sorted(agg.items(), key=lambda kv: (kv[0][1], kv[0][3], kv[0][2])):
        tops = 2 * M * N * K * n / (t * 1e-3) / 1e12
        print(f"  {tag:7s} M={M:4d} K={K:5d} N={N:5d}: {t/n*1000:7.1f} us/call ({tops:5.1f} T) x{n//R} per frame = {t/R:6.2f} ms/frame")
        tot += t / R
    print(f"  total GEMM time per frame {tot:.1f} ms")
    del fast; torch.cuda.empty_cache()
