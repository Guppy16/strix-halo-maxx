"""Evaluate a FastMD2 configuration on the test set: accuracy + latency + agreement with baseline.
usage: evalfast.py LABEL [--dtype bf16|fp16] [--graph] [--limit N] [--compile]"""
import sys, os, torch, json, argparse, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import md2fast
from md2fast import FastMD2
ap = argparse.ArgumentParser(); ap.add_argument("label"); ap.add_argument("--dtype", default="bf16"); ap.add_argument("--graph", action="store_true")
ap.add_argument("--limit", type=int, default=None); ap.add_argument("--lin", default="Lin"); ap.add_argument("--note", default="")
ap.add_argument("--nolog", action="store_true"); ap.add_argument("--tunable", action="store_true"); ap.add_argument("--blas", default="default"); ap.add_argument("--compile", default=""); ap.add_argument("--pool", type=int, default=0); ap.add_argument("--tlin", default=""); ap.add_argument("--nofuse", action="store_true"); ap.add_argument("--fold", action="store_true"); ap.add_argument("--nofusedres", action="store_true"); ap.add_argument("--nq", type=int, default=1)
args = ap.parse_args()
dt = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
if args.blas != "default": torch.backends.cuda.preferred_blas_library(args.blas)
if args.tunable:
    torch.cuda.tunable.enable(True); torch.cuda.tunable.tuning_enable(True)
    torch.cuda.tunable.set_filename(os.path.expanduser(f"~/moondream-opt/tunable_{args.dtype}_{args.blas}.csv"))
model = load_model()
import tgemm
def _cls(n): return getattr(md2fast, n) if hasattr(md2fast, n) else getattr(tgemm, n)
lin_cls = _cls(args.lin); tlin_cls = _cls(args.tlin) if args.tlin else None
fast = FastMD2(model, dtype=dt, lin_cls=lin_cls, pool=args.pool, tlin_cls=tlin_cls, fuse=not args.nofuse, fold_proj=args.fold, fused_res=not args.nofusedres, n_q=args.nq)
if args.compile:
    fast._encode = torch.compile(fast._encode, mode=args.compile, dynamic=False)
    fast._query = torch.compile(fast._query, mode=args.compile, dynamic=False)
# warmup / tuning
for it in items[:3] if False else load_testset(3):
    fast.answer(prep(it["path"]), question(it["obj"]))
if args.tunable:
    torch.cuda.tunable.tuning_enable(False); print("tunable: tuning done", flush=True)
if args.graph: fast.capture()
items = load_testset(args.limit)
base = json.load(open(os.path.expanduser("~/moondream-opt/ref/baseline.json")))["preds"] if os.path.exists(os.path.expanduser("~/moondream-opt/ref/baseline.json")) else {}
ref_emb = torch.load(os.path.expanduser("~/moondream-opt/ref/baseline_emb.pt")) if os.path.exists(os.path.expanduser("~/moondream-opt/ref/baseline_emb.pt")) else {}
# warmup
for it in items[:3]:
    fast.answer(prep(it["path"]), question(it["obj"]))
st = Stats(); T = Timer(); coss = []; margins = []
for i, it in enumerate(items):
    im = prep(it["path"]); q = question(it["obj"])
    fast.set_image(im); fast.set_question(q)
    T.start(); fast.encode(); enc = T.lap(); lg = fast.query(); qry = T.lap()
    pred, tok, m = fast.decide(lg)
    st.add(it, pred, enc, qry); margins.append(m)
    if it["path"] in ref_emb and not args.graph:
        with torch.inference_mode(): e = fast._vision(fast.img_buf).float().cpu()
        coss.append(F.cosine_similarity(e.flatten(), ref_emb[it["path"]].flatten(), dim=0).item())
    if i % 100 == 0: print(i, pred, f"{m:+.2f}", f"{enc:.1f}+{qry:.1f} ms", flush=True)
agree, n = agreement(st.preds(), base)
s0 = fast.answer(prep(f"{SESS}/sessA/judged_0060.jpg"), question("mug")); s1 = fast.answer(prep(f"{SESS}/sessA/judged_0000.jpg"), question("mug"))
line = f"{st.fmt()} | agree w/ baseline {agree*100:.1f}% (n={n}) | emb cos min {min(coss) if coss else float('nan'):.5f} | sanity 0060={s0[0]} 0000={s1[0]}"
print(f"[{args.label}] {line}", flush=True)
if not args.nolog:
    log_result(f"\n## {args.label}\nconfig: dtype={args.dtype} graph={args.graph} lin={args.lin} tlin={args.tlin} blas={args.blas} tunable={args.tunable} compile={args.compile} pool={args.pool} fuse={not args.nofuse} fold={args.fold} fusedres={not args.nofusedres} nq={args.nq} {args.note}\n{line}")
json.dump({"preds": st.preds(), "summary": st.summary(), "margins": margins}, open(os.path.expanduser(f"~/moondream-opt/ref/{args.label}.json"), "w"))
