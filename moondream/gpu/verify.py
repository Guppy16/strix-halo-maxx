"""Check FastMD2 (eager + graph) against the stock path: sanity gate, cosine of img_emb, logits."""
import sys, os, torch, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from md2fast import FastMD2, PREFIX
import md2fast, tgemm, hgemm
def cls(n):
    for mod in (md2fast, tgemm, hgemm):
        if hasattr(mod, n): return getattr(mod, n)
    raise KeyError(n)
vl, tl = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("Lin", "Lin")   # e.g. verify.py HLin HI8Lin
model = load_model(); mm = model.model
fast = FastMD2(model, lin_cls=cls(vl), tlin_cls=cls(tl)); fast.full_logits = True
print(f"fast path linears: vision={vl} text={tl}")
print("yes ids", fast.yes_ids, [mm.tokenizer.decode([i]) for i in fast.yes_ids], "no ids", fast.no_ids, [mm.tokenizer.decode([i]) for i in fast.no_ids])
paths = [f"{SESS}/sessA/judged_0060.jpg", f"{SESS}/sessA/judged_0000.jpg"]
ref_emb = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref/baseline_emb.pt")) if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref/baseline_emb.pt")) else {}
paths += list(ref_emb.keys())[:6]
# stock reference for logits: run encode_image + _prefill_prompt manually
def stock(im, q):
    e = model.encode_image(im); mm.load_encoded_image(e)
    t = mm.config.tokenizer.templates["query"]
    ids = t["prefix"] + mm.tokenizer.encode(q).ids + t["suffix"] + t["suffix"]
    with torch.inference_mode():
        logits, _, nt, _ = mm._prefill_prompt(torch.tensor([ids], device="cuda"), e.pos, 0.0, 0.3)
    return logits[0].float(), int(nt.item())
orig = mm._vis_proj; last = {}
def hook(g, r):
    o = orig(g, r); last["e"] = o; return o
mm._vis_proj = hook
for mode in ("eager", "graph"):
    if mode == "graph": fast.capture()
    for p in paths:
        im = prep(p); q = question("mug")
        lg_s, tok_s = stock(im, q); emb_s = last["e"].float()
        with torch.inference_mode():
            fast.set_image(im); fast.set_question(q); fast.encode()
            emb_f = fast._vision(fast.img_buf).float() if mode == "eager" else None
            lg_f = fast.query()[0].float()
        pred, tok_f, margin = fast.decide(lg_f[None])
        cos = F.cosine_similarity(emb_f.flatten(), emb_s.flatten(), dim=0).item() if emb_f is not None else float("nan")
        cosref = F.cosine_similarity(emb_s.flatten().cpu(), ref_emb[p].flatten(), dim=0).item() if p in ref_emb else float("nan")
        top_s = mm.tokenizer.decode([tok_s]); top_f = mm.tokenizer.decode([tok_f])
        print(f"[{mode}] {p.split('/')[-2]}/{p.split('/')[-1]}: stock top={top_s!r} fast top={top_f!r} yes={pred} margin={margin:+.2f} | logits maxabsdiff={(lg_s-lg_f).abs().max():.3f} cos(logits)={F.cosine_similarity(lg_s,lg_f,dim=0):.5f} | emb cos vs stock={cos:.6f} stock-vs-ref={cosref:.6f}", flush=True)
