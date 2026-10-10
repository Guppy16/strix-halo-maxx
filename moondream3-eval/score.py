"""Grade detect/point results with the user's verdicts (data/box_labels.jsonl), the same way
.scratch/grounding/replay_hybrid.py does: the first box takes the verdict of the labelled
answer it overlaps (IoU > 0.5, best wins); absent -> correct iff nothing returned.
Points: the verdict of the smallest labelled box that contains the point.

  python score.py results/md2_detect.jsonl results/md3_detect.jsonl ...
Each row: {"id", "boxes": [[x1,y1,x2,y2],...]} or {"id", "points": [[x,y],...]}, "ms".
"""

import json
import os
import statistics
import sys

DATA = os.path.expanduser("~/git/app/data")


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_truth():
    queue = {q["id"]: q for q in json.load(open(f"{DATA}/box_queue/grounding-2026-10-09.json"))["items"]}
    labels = {}
    for line in open(f"{DATA}/box_labels.jsonl"):
        r = json.loads(line)
        labels[r["key"].split("/")[-1]] = r
    return queue, labels


def labelled_boxes(q, lab):
    """(box, verdict) for every labelled answer box of the query."""
    out = []
    for model, ans in q["answers"].items():
        v = lab["verdicts"].get(model)
        if v:
            out += [(b, v) for b in ans.get("boxes", [])]
    return out


def grade(row, q, lab) -> str:
    if lab is None or (not lab["absent"] and not lab["verdicts"]):
        return "skip"
    found = row.get("boxes") or row.get("points")
    if lab["absent"]:
        return "correct" if not found else "wrong"
    if not found:
        return "missed"
    if "boxes" in row:
        best, verdict = 0.5, "unlabelled"
        for b, v in labelled_boxes(q, lab):
            if iou(row["boxes"][0], b) > best:
                best, verdict = iou(row["boxes"][0], b), v
        return verdict
    x, y = row["points"][0]
    inside = [(b, v) for b, v in labelled_boxes(q, lab) if b[0] <= x <= b[2] and b[1] <= y <= b[3]]
    if not inside:
        return "unlabelled"
    return min(inside, key=lambda bv: (bv[0][2] - bv[0][0]) * (bv[0][3] - bv[0][1]))[1]


def score(path):
    queue, labels = load_truth()
    rows = [json.loads(line) for line in open(path)]
    out = {}
    for r in rows:
        out[r["id"]] = grade(r, queue[r["id"]], labels.get(r["id"]))
    tally = {}
    for v in out.values():
        tally[v] = tally.get(v, 0) + 1
    ms = [r["ms"] for r in rows if "ms" in r]
    return out, tally, ms


if __name__ == "__main__":
    for p in sys.argv[1:]:
        _, tally, ms = score(p)
        graded = sum(v for k, v in tally.items() if k != "skip")
        med = statistics.median(ms) if ms else 0
        p90 = sorted(ms)[int(0.9 * len(ms))] if ms else 0
        print(f"{os.path.basename(p):24s} {tally.get('correct', 0)}/{graded} correct  {tally}  "
              f"median {med:.0f} ms p90 {p90:.0f} ms")
