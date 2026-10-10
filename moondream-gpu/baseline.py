"""Baseline: stock HF moondream2 bf16, model.encode_image + model.query (temperature 0).
Records per-frame predictions, latencies, and reference image embeddings for cosine checks."""
import sys, json, os, torch, statistics, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
model = load_model()
mm = model.model
items = load_testset(limit)
print(f"{len(items)} frames", flush=True)

# capture the vision projection output for reference
ref_emb = {}
orig_vis_proj = mm._vis_proj
_last = {}
def hook(g, r):
    out = orig_vis_proj(g, r); _last["emb"] = out; return out
mm._vis_proj = hook

# warmup
im = prep(items[0]["path"])
for _ in range(3):
    e = model.encode_image(im); model.query(e, question("mug"), settings={"temperature": 0, "max_tokens": 4})

st = Stats(); T = Timer()
raw = {}
keep_emb = set(it["path"] for it in items[::16])
for i, it in enumerate(items):
    im = prep(it["path"])
    T.start()
    e = model.encode_image(im); enc = T.lap()
    ans = model.query(e, question(it["obj"]), settings={"temperature": 0, "max_tokens": 4})["answer"]; q = T.lap()
    st.add(it, parse_yes(ans), enc, q)
    raw[it["path"]] = ans
    if it["path"] in keep_emb:
        ref_emb[it["path"]] = _last["emb"].detach().float().cpu()
    if i % 50 == 0:
        print(i, it["path"].split("/")[-2:], repr(ans), f"{enc:.0f}+{q:.0f} ms", flush=True)

print(st.fmt(), flush=True)
os.makedirs(os.path.expanduser("~/moondream-opt/ref"), exist_ok=True)
json.dump({"preds": st.preds(), "raw": raw, "summary": st.summary()}, open(os.path.expanduser("~/moondream-opt/ref/baseline.json"), "w"), indent=1)
torch.save(ref_emb, os.path.expanduser("~/moondream-opt/ref/baseline_emb.pt"))
# sanity gate
s0 = prep(f"{SESS}/sessA/judged_0060.jpg"); s1 = prep(f"{SESS}/sessA/judged_0000.jpg")
a0 = model.query(s0, question("mug"), settings={"temperature": 0, "max_tokens": 4})["answer"]
a1 = model.query(s1, question("mug"), settings={"temperature": 0, "max_tokens": 4})["answer"]
print("sanity: 0060 mug ->", repr(a0), "| 0000 mug ->", repr(a1))
log_result(f"\n## Baseline (stock HF bf16, 378x378 squash, encode_image+query temp0)\n{st.fmt()}\nsanity: 0060 mug -> {a0!r}, 0000 mug -> {a1!r}\nraw answer histogram: { {k: list(raw.values()).count(k) for k in set(raw.values())} }")
