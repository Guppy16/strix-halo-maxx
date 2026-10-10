"""moondream 0.5B (official int8 .mf, moondream==0.0.6 client, onnxruntime CPU) on the moondream2 test set.
usage: venv05/bin/python eval05.py LABEL [--limit N] [--threads T] [--probe] [--nolog]
Reuses common.py (test set, prep, metrics). common.py imports torch (not in venv05) -> stubbed."""
import sys, os, types, json, time, statistics, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
_t = types.ModuleType("torch"); _nn = types.ModuleType("torch.nn"); _F = types.ModuleType("torch.nn.functional")
_t.nn = _nn; _nn.functional = _F; _t.bfloat16 = None; _t.cuda = types.SimpleNamespace(synchronize=lambda: None)
sys.modules.update({"torch": _t, "torch.nn": _nn, "torch.nn.functional": _F})
from common import load_testset, prep, question, parse_yes, Stats, agreement, SESS
import numpy as np, onnxruntime as ort
import moondream.onnx_vl as ovl
from moondream.moonfile import unpack
from moondream.preprocess import create_patches
from tokenizers import Tokenizer

ap = argparse.ArgumentParser(); ap.add_argument("label"); ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--threads", type=int, default=16); ap.add_argument("--probe", action="store_true"); ap.add_argument("--nolog", action="store_true")
ap.add_argument("--model", default=os.path.join(HERE, "models/moondream-0_5b-int8.mf")); ap.add_argument("--prep", default="lanczos378")
args = ap.parse_args()

def load(path, threads):
    """Same as OnnxVL.from_path (CPU branch) but with a thread limit (shared box)."""
    ort.set_default_logger_severity(3)
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    comps = {}
    for fn, c in unpack(path):
        key, ext = fn.split("/")[-1].split(".")
        if key == "tokenizer": comps[key] = Tokenizer.from_buffer(c)
        elif ext == "onnx": comps[key] = ort.InferenceSession(c, sess_options=so, providers=["CPUExecutionProvider"])
        elif ext == "json": comps[key] = json.loads(c)
        elif ext == "npy": comps[key] = np.load(__import__("io").BytesIO(c))
    return ovl.OnnxVL(special_tokens=comps["config"]["special_tokens"], templates=comps["config"]["templates"], **comps)

t0 = time.time(); m = load(args.model, args.threads); print(f"loaded in {time.time()-t0:.1f}s, ort {ort.__version__}, providers {m.text_decoder.get_providers()}", flush=True)
tok = m.tokenizer
ids = lambda s: tok.encode(s).ids
yes_ids = sorted({ids(s)[0] for s in ("Yes", "yes", " Yes", " yes") if len(ids(s)) == 1})
no_ids = sorted({ids(s)[0] for s in ("No", "no", " No", " no") if len(ids(s)) == 1})
print("yes ids", yes_ids, "no ids", no_ids, flush=True)

def prompt_embeds(q):
    toks = m.templates["query"]["prefix"] + ids(q) + m.templates["query"]["suffix"]
    (e,) = m.text_encoder.run(None, {"input_ids": [toks]}); return e, len(toks)

def yn_logits(enc, q):
    """First decode step only (prompt prefill over the image KV cache) -> logits of the next token."""
    e, n = prompt_embeds(q)
    kv = enc.kv_cache  # already exactly pos long (no 2048 copy needed for a single prefill)
    (lg,) = m.text_decoder.run(["logits"], {"input_embeds": e, "kv_cache": kv})
    lg = np.asarray(lg, dtype=np.float32).reshape(-1)
    return lg

def decide(lg):
    y = max(lg[i] for i in yes_ids); n = max(lg[i] for i in no_ids); return bool(y > n), float(y - n), int(np.argmax(lg))

def prep_img(path):
    if args.prep == "lanczos378": return prep(path)          # identical to the moondream2 harness (378x378 LANCZOS squash -> single tile)
    from PIL import Image; return Image.open(path).convert("RGB")  # raw 720x1280 -> client picks the (2,1) multi-crop template

if args.probe:
    cfg = dict(m.config); print("config:", json.dumps({k: v for k, v in cfg.items() if k != "templates"})[:1500], flush=True)
    print("templates.query:", m.templates["query"], flush=True)
    print("initial_kv_cache:", m.initial_kv_cache.shape, m.initial_kv_cache.dtype, flush=True)
    for name in ("vision_encoder", "vision_projection", "text_encoder", "text_decoder"):
        s = getattr(m, name); print(name, [(i.name, i.type, i.shape) for i in s.get_inputs()], "->", [(o.name, o.type, o.shape) for o in s.get_outputs()], flush=True)
    for f in (f"{SESS}/sessA/judged_0060.jpg", f"{SESS}/sessA/judged_0000.jpg"):
        im = prep_img(f); patches, tmpl = create_patches(im); print(f.split('/')[-1], "patches", patches.shape, patches.dtype, "template", tmpl, flush=True)
        for r in range(3):
            t0 = time.perf_counter(); enc = m.encode_image(im); t1 = time.perf_counter()
            lg = yn_logits(enc, question("mug")); t2 = time.perf_counter()
            ans = m.query(enc, question("mug"), settings={"max_tokens": 4})["answer"]; t3 = time.perf_counter()
            p, mg, top = decide(lg)
            print(f"  run{r}: pos={enc.pos} kv={enc.kv_cache.shape} {enc.kv_cache.dtype} | encode {1000*(t1-t0):.0f} ms, yn-logits {1000*(t2-t1):.0f} ms, greedy query {1000*(t3-t2):.0f} ms | logits yes={p} margin={mg:+.2f} top={top!r}={tok.decode([top])!r} | answer={ans!r}", flush=True)
    sys.exit(0)

items = load_testset(args.limit)
for it in items[:2]:  # warmup
    enc = m.encode_image(prep_img(it["path"])); yn_logits(enc, question(it["obj"])); m.query(enc, question(it["obj"]), settings={"max_tokens": 4})
st_l = Stats(); st_g = Stats(); margins = {}; raw = {}; tq = []
t_start = time.time()
for i, it in enumerate(items):
    im = prep_img(it["path"]); q = question(it["obj"])
    t0 = time.perf_counter(); enc = m.encode_image(im); t1 = time.perf_counter()
    lg = yn_logits(enc, q); t2 = time.perf_counter()
    ans = m.query(enc, q, settings={"max_tokens": 4})["answer"]; t3 = time.perf_counter()
    p, mg, top = decide(lg)
    st_l.add(it, p, 1000 * (t1 - t0), 1000 * (t2 - t1)); st_g.add(it, parse_yes(ans), 1000 * (t1 - t0), 1000 * (t3 - t2))
    margins[it["path"]] = mg; raw[it["path"]] = ans
    if i % 50 == 0: print(i, it["obj"], it["label"], p, f"{mg:+.2f}", repr(ans), f"{1000*(t1-t0):.0f}+{1000*(t2-t1):.0f}/{1000*(t3-t2):.0f} ms", f"elapsed {time.time()-t_start:.0f}s", flush=True)
base = json.load(open(os.path.join(HERE, "ref/baseline.json")))["preds"]
opt = json.load(open(os.path.join(HERE, "ref/fused-nofold-nofusedres-int8txt-graph.json")))["preds"]
al, nl = agreement(st_l.preds(), base); ag, ng = agreement(st_g.preds(), base); ao, _ = agreement(st_l.preds(), opt); alg, _ = agreement(st_l.preds(), st_g.preds())
hist = {k: list(raw.values()).count(k) for k in set(raw.values())}
s0 = decide(yn_logits(m.encode_image(prep_img(f"{SESS}/sessA/judged_0060.jpg")), question("mug")))[0]
s1 = decide(yn_logits(m.encode_image(prep_img(f"{SESS}/sessA/judged_0000.jpg")), question("mug")))[0]
line = (f"logits yes/no: {st_l.fmt()} | agree w/ stock md2 {al*100:.1f}% (n={nl}), w/ optimised md2 {ao*100:.1f}%\n"
        f"greedy text (max_tokens=4): {st_g.fmt()} | agree w/ stock md2 {ag*100:.1f}% | logits-vs-greedy agreement {alg*100:.1f}% | raw answers {hist}\n"
        f"sanity (logits) 0060 mug={s0} 0000 mug={s1} | threads={args.threads} prep={args.prep}")
print(f"[{args.label}]\n{line}", flush=True)
json.dump({"preds": st_l.preds(), "preds_greedy": st_g.preds(), "raw": raw, "margins": margins, "summary": st_l.summary(), "summary_greedy": st_g.summary()},
          open(os.path.join(HERE, f"ref/{args.label}.json"), "w"))
if not args.nolog:
    with open(os.path.join(HERE, "RESULTS-0.5b.md"), "a") as f: f.write(f"\n## {args.label}\n{line}\n")
