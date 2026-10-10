"""Frames where any model's answer differs from the judged label (data/calibration/frame_preds.json)."""

import glob
import json
import re

preds = json.load(open("data/calibration/frame_preds.json"))
for f in sorted(glob.glob("data/sessions/*/judged.jsonl")):
    sid = f.split("/")[2]
    for line in open(f):
        r = json.loads(line)
        m = re.search(r"(mug|tv)", r["task"])
        if not m:
            continue
        key, label = f"{sid}/{r['file']}", r["status"] == "found"
        answers = {name: p[key] for name, p in preds.items()}
        if any(a["yes"] != label for a in answers.values()):
            cols = "  ".join(f"{n}={'Y' if a['yes'] else 'n'}({a['margin']:+.1f})" for n, a in answers.items())
            print(f"{key:32} {m.group(1):3} label={'yes' if label else 'no '}  {cols}")
