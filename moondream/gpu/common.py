"""Shared helpers: model loading, test set, metrics, timing."""
import glob, json, os, re, statistics, time
import torch.nn.functional as F
import torch
from PIL import Image

SESS = os.environ.get("MD_SESS", os.path.expanduser("~/git/app/data/sessions"))
REV = "2025-06-21"
OBJ_RE = re.compile(r"(mug|tv)")
QUESTION = "Is there a {obj} in this image? Answer yes or no."


def load_model(dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2", revision=REV, trust_remote_code=True,
        dtype=dtype, device_map={"": "cuda"}).eval()
    m._setup_caches()
    return m


def load_testset(limit=None):
    """Returns list of dicts: path, task, obj, label(bool)."""
    items = []
    for f in sorted(glob.glob(f"{SESS}/*/judged.jsonl")):
        d = os.path.dirname(f)
        for line in open(f):
            r = json.loads(line)
            m = OBJ_RE.search(r["task"])
            if not m:
                continue
            items.append(dict(path=os.path.join(d, r["file"]), task=r["task"], obj=m.group(1),
                              label=(r["status"] == "found"), session=os.path.basename(d)))
    if limit:
        items = items[:limit]
    return items


def prep(path, size=378, draft=True):
    """Baseline preprocessing: squash the frame to size x size with LANCZOS (matches what the model
    does internally for any single-tile image). draft=True lets libjpeg decode the 720x1280 JPEG at
    1/2 scale first (DCT-domain), which is bit-identical after the LANCZOS resize on all test frames
    (verified: max abs pixel diff 0 over 99 frames) and ~2.5x faster (~3 ms instead of ~8 ms)."""
    im = Image.open(path)
    if draft and im.format == "JPEG":
        im.draft("RGB", (size * 2, size * 2))
    return im.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def question(obj):
    return QUESTION.format(obj=obj)


def parse_yes(ans: str):
    a = ans.strip().lower()
    return a.startswith("yes")


class Stats:
    def __init__(self):
        self.rows = []  # (item, pred_yes, enc_ms, qry_ms)

    def add(self, item, pred, enc_ms, qry_ms):
        self.rows.append((item, pred, enc_ms, qry_ms))

    def summary(self):
        out = {}
        by = {}
        for it, p, e, q in self.rows:
            by.setdefault(it["obj"], []).append((it["label"], p))
        allc = 0; alln = 0
        for obj, lst in by.items():
            n = len(lst); c = sum(1 for l, p in lst if l == p)
            tp = sum(1 for l, p in lst if l and p); fn = sum(1 for l, p in lst if l and not p)
            fp = sum(1 for l, p in lst if (not l) and p); tn = sum(1 for l, p in lst if (not l) and (not p))
            out[obj] = dict(n=n, acc=c / n, tp=tp, fn=fn, fp=fp, tn=tn)
            allc += c; alln += n
        out["all"] = dict(n=alln, acc=allc / alln)
        enc = [e for _, _, e, _ in self.rows if e is not None]
        qry = [q for _, _, _, q in self.rows if q is not None]
        if enc:
            out["enc_ms"] = statistics.median(enc); out["qry_ms"] = statistics.median(qry)
            out["tot_ms"] = statistics.median([e + q for e, q in zip(enc, qry)])
        return out

    def fmt(self):
        s = self.summary()
        parts = [f"{k}: acc={v['acc']*100:.1f}% n={v['n']}" + (f" tp={v['tp']} fn={v['fn']} fp={v['fp']} tn={v['tn']}" if 'tp' in v else "")
                 for k, v in s.items() if isinstance(v, dict)]
        if "enc_ms" in s:
            parts.append(f"median encode {s['enc_ms']:.1f} ms, query {s['qry_ms']:.1f} ms, total {s['tot_ms']:.1f} ms")
        return " | ".join(parts)

    def preds(self):
        return {r[0]["path"]: r[1] for r in self.rows}


def agreement(preds_a, preds_b):
    keys = [k for k in preds_a if k in preds_b]
    return sum(1 for k in keys if preds_a[k] == preds_b[k]) / max(1, len(keys)), len(keys)


class Timer:
    def __init__(self):
        self.t = None

    def start(self):
        torch.cuda.synchronize(); self.t = time.perf_counter()

    def lap(self):
        torch.cuda.synchronize(); n = time.perf_counter(); d = (n - self.t) * 1000; self.t = n; return d


def log_result(text):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RESULTS.md")
    with open(p, "a") as f:
        f.write(text.rstrip() + "\n")
