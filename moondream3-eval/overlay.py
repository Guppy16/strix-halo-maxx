"""Draw md2 (orange) and md3 (cyan) detect boxes + labelled boxes (green correct, red wrong)."""
import json
import sys

from PIL import Image, ImageDraw

from score import DATA, load_truth

queue, labels = load_truth()
rows = {m: {json.loads(l)["id"]: json.loads(l) for l in open(f"results/{m}_detect.jsonl")} for m in ("md2", "md3")}
pts = {m: {json.loads(l)["id"]: json.loads(l) for l in open(f"results/{m}_point.jsonl")} for m in ("md2", "md3")}
for qid in sys.argv[1:]:
    q = queue[qid]
    img = Image.open(f"{DATA}/{q['frame']}").convert("RGB")
    W, H = img.size
    d = ImageDraw.Draw(img)
    lab = labels.get(qid) or {'verdicts': {}}
    for model, ans in q["answers"].items():
        v = lab['verdicts'].get(model)
        for b in ans.get("boxes", []):
            if v:
                d.rectangle([b[0] * W, b[1] * H, b[2] * W, b[3] * H], outline="lime" if v == "correct" else "red", width=2)
    for m, col in (("md2", "orange"), ("md3", "cyan")):
        for b in rows[m][qid]["boxes"]:
            d.rectangle([b[0] * W, b[1] * H, b[2] * W, b[3] * H], outline=col, width=4)
        for x, y in pts[m][qid]["points"]:
            d.ellipse([x * W - 9, y * H - 9, x * W + 9, y * H + 9], outline=col, width=3)
    d.text((10, 10), f"{qid} {q['object']}", fill="yellow")
    img.save(f"out/{qid}.jpg", quality=85)
