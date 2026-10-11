"""Prompt-format / preprocessing probe for moondream 0.5B: is the black-frame 'Yes' a prompt artefact?"""
import sys, os, types, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
_t = types.ModuleType("torch"); _nn = types.ModuleType("torch.nn"); _F = types.ModuleType("torch.nn.functional")
_t.nn = _nn; _nn.functional = _F; _t.bfloat16 = None; _t.cuda = types.SimpleNamespace(synchronize=lambda: None)
sys.modules.update({"torch": _t, "torch.nn": _nn, "torch.nn.functional": _F})
from common import load_testset, prep, question, SESS
import numpy as np
sys.argv = ["x", "probe", "--probe"]  # reuse eval05's loader without running its main
import importlib.util
spec = importlib.util.spec_from_file_location("e5", os.path.join(HERE, "eval05.py")); src = open(os.path.join(HERE, "eval05.py")).read().split("if args.probe:")[0]
g = {"__name__": "e5", "__file__": os.path.join(HERE, "eval05.py")}; exec(compile(src, "eval05.py", "exec"), g)
m, decide, yn_logits, tok = g["m"], g["decide"], g["yn_logits"], g["tok"]
from PIL import Image
items = load_testset()
found = [it for it in items if it["label"]][:8]; notf = [it for it in items if not it["label"]][::40][:8]
sel = [dict(path=f"{SESS}/sessA/judged_0060.jpg", obj="mug", label=True, tag="SANITY-0060"), dict(path=f"{SESS}/sessA/judged_0000.jpg", obj="mug", label=False, tag="SANITY-0000-black")] + found + notf
prompts = {"std": "Is there a {obj} in this image? Answer yes or no.", "bare": "Is there a {obj} in this image?", "contain": "Does this image contain a {obj}? Answer yes or no.",
           "see": "Can you see a {obj}? Answer yes or no.", "neg": "Is there a giraffe in this image? Answer yes or no."}
print("tag/session obj label | " + " | ".join(f"{k}: yes(margin)" for k in prompts) + " | caption(short)", flush=True)
hits = {k: 0 for k in prompts}; n = 0
for it in sel:
    enc = m.encode_image(prep(it["path"]))
    cells = []
    for k, p in prompts.items():
        y, mg, top = decide(yn_logits(enc, p.format(obj=it["obj"])))
        cells.append(f"{k}: {'Y' if y else 'n'}({mg:+.2f})")
        if k != "neg": hits[k] += int(y == it["label"])
    n += 1
    cap = m.caption(enc, length="short", settings={"max_tokens": 20})["caption"].strip().replace("\n", " ")
    print(f"{(it.get('tag') or it['session'])} {it['obj']} {it['label']} | " + " | ".join(cells) + f" | {cap!r}", flush=True)
print("acc on this subset:", {k: f"{v}/{n}" for k, v in hits.items() if k != "neg"}, flush=True)
# raw 720x1280 (client's own 2x1 multi-crop) on the sanity frames
for f in (f"{SESS}/sessA/judged_0060.jpg", f"{SESS}/sessA/judged_0000.jpg"):
    im = Image.open(f).convert("RGB"); t0 = time.perf_counter(); enc = m.encode_image(im); t1 = time.perf_counter()
    y, mg, top = decide(yn_logits(enc, question("mug")))
    print(f"raw multi-crop {f.split('/')[-1]}: pos={enc.pos} encode {1000*(t1-t0):.0f} ms -> mug yes={y} margin={mg:+.2f}", flush=True)
