#!/usr/bin/env python
"""End-to-end benchmark of the optimised moondream2 yes/no path (and optionally the stock path) on the test set.

  venv-rocm714/bin/python bench.py         # best exact config: HIP WMMA bf16 vision + HIP WMMA int8 text (hgemm.py), fused kernels, HIP graphs
  venv-rocm714/bin/python bench.py --vlin TLin --tlin I8Lin   # previous best (Triton GEMMs)
  venv/bin/python bench.py --pool 16       # "fast mode": image tokens avg-pooled 27x27 -> 16x16 before the LLM (see RESULTS.md)
  venv/bin/python bench.py --stock         # also run the stock HF path in the same process for comparison
  venv/bin/python bench.py --image f.jpg --question "Is there a mug in this image? Answer yes or no."
"""
import argparse, json, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from common import load_model, load_testset, prep, question, parse_yes, Stats, Timer, agreement, SESS
import md2fast, tgemm, hgemm
from md2fast import FastMD2

ap = argparse.ArgumentParser()
ap.add_argument("--pool", type=int, default=0, help="0 = exact 729 image tokens; k = avg-pool to kxk tokens (16 recommended)")
ap.add_argument("--vlin", default="HLin", help="vision linear impl: HLin (HIP WMMA bf16) | TLin (triton bf16) | Lin (rocBLAS) | I8Lin")
ap.add_argument("--tlin", default="HI8Lin", help="text linear impl: HI8Lin (HIP WMMA W8A8) | I8Lin (triton W8A8) | TLin | Lin")
ap.add_argument("--nograph", action="store_true"); ap.add_argument("--nofuse", action="store_true")
ap.add_argument("--hattn", type=int, default=0, help="1 = HIP WMMA flash attention for the two mask-free prefills (hattn.py), 0 = torch SDPA")
ap.add_argument("--nq", type=int, default=1, help="questions per frame (batched prompt prefill)")
ap.add_argument("--frames", type=int, default=None, help="limit number of test frames")
ap.add_argument("--stock", action="store_true", help="also benchmark the stock HF encode_image+query path")
ap.add_argument("--image", default=None); ap.add_argument("--question", default=None)
args = ap.parse_args()

def cls(n):
    for mod in (md2fast, tgemm, hgemm):
        if hasattr(mod, n): return getattr(mod, n)
    raise KeyError(n)
t0 = time.time(); model = load_model(); print(f"model loaded in {time.time()-t0:.1f}s", flush=True)
t0 = time.time()
fast = FastMD2(model, lin_cls=cls(args.vlin), tlin_cls=cls(args.tlin), pool=args.pool, fuse=not args.nofuse, n_q=args.nq, hattn=bool(args.hattn))
objs = ["mug", "tv"]
qs = [question(objs[i % len(objs)]) for i in range(args.nq)]
for _ in range(3):  # warmup + triton autotune
    fast.set_image(prep(f"{SESS}/sessA/judged_0060.jpg")); fast.set_questions(qs); fast.encode(); fast.query()
if not args.nograph: fast.capture()
torch.cuda.synchronize(); print(f"fast path built (autotune+capture) in {time.time()-t0:.1f}s", flush=True)

if args.image:
    im = prep(args.image); q = args.question or question("mug")
    fast.set_image(im); fast.set_questions([q] * args.nq)
    T = Timer(); T.start(); fast.encode(); e = T.lap(); lg = fast.query(); qq = T.lap()
    pred, _, m = fast.decide(lg)
    print(f"{args.image}: {q} -> {'yes' if pred else 'no'} (margin {m:+.2f}) | encode {e:.1f} ms, query {qq:.1f} ms, total {e+qq:.1f} ms")
    sys.exit(0)

items = load_testset(args.frames)
st = Stats(); T = Timer()
for it in items:
    im = prep(it["path"]); fast.set_image(im); fast.set_questions([question(it["obj"])] * args.nq)
    T.start(); fast.encode(); e = T.lap(); lg = fast.query(); qq = T.lap()
    st.add(it, fast.decide(lg)[0], e, qq)
s0 = fast.answer(prep(f"{SESS}/sessA/judged_0060.jpg"), question("mug"))[0]; s1 = fast.answer(prep(f"{SESS}/sessA/judged_0000.jpg"), question("mug"))[0]
gate = f"sanity gate: judged_0060 mug={'yes' if s0 else 'NO (FAIL)'}, judged_0000 mug={'no' if not s0 or not s1 else 'YES (FAIL)'}"
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref/baseline.json")
extra = ""
if os.path.exists(base_path):
    b = json.load(open(base_path)); a, n = agreement(st.preds(), b["preds"]); extra = f" | agreement with stored stock baseline {a*100:.1f}% (n={n}); stored baseline: {b['summary']['all']['acc']*100:.1f}% acc, {b['summary']['tot_ms']:.0f} ms"
print(f"\nFAST [{args.vlin}/{args.tlin} hattn={args.hattn} pool={args.pool} graph={not args.nograph} fuse={not args.nofuse} nq={args.nq}]:\n  {st.fmt()}\n  {gate}{extra}", flush=True)

if args.stock:
    mm = model.model; st2 = Stats()
    im = prep(items[0]["path"])
    for _ in range(3): model.query(model.encode_image(im), question("mug"), settings={"temperature": 0, "max_tokens": 4})
    for it in items:
        im = prep(it["path"]); T.start(); e = model.encode_image(im); em = T.lap()
        ans = model.query(e, question(it["obj"]), settings={"temperature": 0, "max_tokens": 4})["answer"]; qm = T.lap()
        st2.add(it, parse_yes(ans), em, qm)
    a, n = agreement(st.preds(), st2.preds())
    print(f"\nSTOCK (HF bf16 encode_image + query, temp 0):\n  {st2.fmt()}\n  fast/stock agreement {a*100:.1f}% (n={n})")
