"""Per-query verdicts for md2 vs md3 (detect and point); prints every disagreement."""
import json
import sys

from score import load_truth, score

queue, labels = load_truth()
for kind in ("detect", "point"):
    a, _, _ = score(f"results/md2_{kind}.jsonl")
    b, _, _ = score(f"results/md3_{kind}.jsonl")
    ra = {json.loads(l)["id"]: json.loads(l) for l in open(f"results/md2_{kind}.jsonl")}
    rb = {json.loads(l)["id"]: json.loads(l) for l in open(f"results/md3_{kind}.jsonl")}
    print(f"== {kind}")
    for qid in a:
        if a[qid] != b[qid] or b[qid] == "unlabelled":
            k = "boxes" if kind == "detect" else "points"
            lab = labels.get(qid)
            print(f"{qid} {queue[qid]['object']!r} absent={lab and lab['absent']} md2={a[qid]} md3={b[qid]}")
            print("   md2", [[round(x, 3) for x in v] for v in ra[qid][k][:3]], " md3", [[round(x, 3) for x in v] for v in rb[qid][k][:3]], "n=", len(rb[qid][k]))
