"""Score the per-frame answers (data/calibration/frame_preds.json) against the user's frame labels
(data/frame_labels.jsonl, last line per frame wins); frames without a user label keep the judge's
label. "unsure" frames and monitors (a grey area: the user marked them "monitor, not a TV") are
left out, so the score is clear TV vs clear not-TV."""

import glob
import json
import re
from collections import Counter
from pathlib import Path

DATA = Path.home() / "git/app/data"
preds = json.loads((DATA / "calibration/frame_preds.json").read_text())
user = {}
for line in (DATA / "frame_labels.jsonl").read_text().splitlines():
    if line.strip():
        lab = json.loads(line)
        user[lab["key"]] = lab

frames = {}
for f in sorted(glob.glob(str(DATA / "sessions/*/judged.jsonl"))):
    sid = Path(f).parent.name
    for line in open(f):
        r = json.loads(line)
        if m := re.search(r"\b(tv|mug)\b", r["task"], re.I):
            frames[f"{sid}/{r['file']}"] = (m.group(1).lower(), r["status"] == "found")

changes = Counter()
truth = {}
for key, (obj, judged) in frames.items():
    lab = user.get(key)
    if lab and lab["present"] == "unsure":
        changes["unsure (left out)"] += 1
        continue
    if lab and "monitor" in lab.get("note", "").lower():
        changes["monitor (left out)"] += 1
        continue
    t = (lab["present"] == "yes") if lab else judged
    if lab:
        changes["judge yes -> no" if judged and not t else "judge no -> yes" if t and not judged else "confirmed"] += 1
    truth[key] = (obj, t)
notes = Counter(lab["note"] for lab in user.values() if lab.get("note"))
print(f"{len(user)} frames labelled: {dict(changes)}; notes: {dict(notes.most_common(5))}")
print(f"scored on {len(truth)} frames ({sum(t for _, t in truth.values())} with the object in view)\n")
print(f"{'config':10} {'judge labels':>22}   {'your labels':>34}")
for name, p in preds.items():
    def score(lab_of):
        c = Counter()
        for key, (obj, _) in truth.items():
            y, t = p[key]["yes"], lab_of(key)
            c["ok" if y == t else "fp" if y else "fn"] += 1
        n = sum(c.values())
        return f"{100 * c['ok'] / n:5.1f}% fp {c['fp']:2} fn {c['fn']:2}"
    print(f"{name:10} {score(lambda k: frames[k][1]):>22}   {score(lambda k: truth[k][1]):>34}")
wrong = [k for k, (o, t) in truth.items() if any(p[k]["yes"] != t for p in preds.values())]
print(f"\nframes some model still gets wrong vs your labels: {len(wrong)}")
for k in wrong:
    o, t = truth[k]
    print(f"  {k:32} {o:3} truth={'yes' if t else 'no '}  " + "  ".join(f"{n}={'Y' if p[k]['yes'] else 'n'}({p[k]['margin']:+.1f})" for n, p in preds.items()))
