"""Per-frame answers of one fast-path config on the test set, for the frame labeller.

Merges {name: {"<session>/<file>": {"yes": bool, "margin": float}}} into
<MD_FRAMES parent>/calibration/frame_preds.json (margin = logsumexp(yes) - logsumexp(no)).

    venv-rocm714/bin/python frame_preds.py --name exact
    venv-rocm714/bin/python frame_preds.py --name pool16 --pool 16
    venv-rocm714/bin/python frame_preds.py --name npu_bfp16 --vlin BfpAcc32Lin
    venv-rocm714/bin/python frame_preds.py --name npu_full --vlin BfpAcc32Lin --bf16 all
"""

import argparse
import json
import os

import torch

import bf16emu
import bfpemu
import hgemm
import md2fast
import tgemm
from common import DATA, SESS, load_model, load_testset, prep, question
from md2fast import FastMD2

OUT = os.path.join(DATA, "calibration", "frame_preds.json")

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--vlin", default="HLin")
ap.add_argument("--tlin", default="HI8Lin")
ap.add_argument("--pool", type=int, default=0)
ap.add_argument("--tv-question", default=None, help="wording for the tv frames (mug frames keep the default)")
ap.add_argument("--bf16", default=None, help="vision ops computed in bf16 only (bf16emu): all or a comma list of ln,gelu,softmax")
args = ap.parse_args()
if args.bf16:
    bf16emu.enable(args.bf16)


def cls(n):
    for mod in (md2fast, tgemm, hgemm, bfpemu):
        if hasattr(mod, n):
            return getattr(mod, n)
    raise KeyError(n)


model = load_model()
fast = FastMD2(model, lin_cls=cls(args.vlin), tlin_cls=cls(args.tlin), pool=args.pool, prompt_len=48)
warm = prep(f"{SESS}/sessA/judged_0060.jpg")
for _ in range(3):
    fast.set_image(warm)
    fast.set_questions([question("mug")])
    fast.encode()
    fast.query()
torch.cuda.synchronize()

preds = {}
for it in load_testset():
    fast.set_image(prep(it["path"]))
    q = args.tv_question if (args.tv_question and it["obj"] == "tv") else question(it["obj"])
    fast.set_questions([q])
    fast.encode()
    yes, _, margin = fast.decide(fast.query())
    preds[f"{it['session']}/{os.path.basename(it['path'])}"] = {"yes": bool(yes), "margin": round(margin, 3)}

all_preds = json.load(open(OUT)) if os.path.exists(OUT) else {}
all_preds[args.name] = preds
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(all_preds, f)
wrong = sum(p["yes"] != it["label"] for it in load_testset() for p in [preds[f"{it['session']}/{os.path.basename(it['path'])}"]])
print(f"{args.name}: {len(preds)} frames, {wrong} differ from the judged labels -> {OUT}")
