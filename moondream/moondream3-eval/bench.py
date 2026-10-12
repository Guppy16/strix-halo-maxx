"""moondream3-preview vs moondream2: detect + point on the 50 labelled grounding queries,
then caption/query latency on 5 frames. One model per process.

  gpu_safe.sh <log> timeout 1800 venv-rocm714/bin/python bench.py md3 [--compile]

-> results/<tag>_detect.jsonl, <tag>_point.jsonl, <tag>_vqa.jsonl, <tag>_summary.json
"""

import argparse
import glob
import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# Private grounding set (not in the repo): MD_FRAMES is the sessions folder; its parent holds the frames
# and grounding_queries.json.
DATA = os.path.dirname(os.path.abspath(os.environ.get("MD_FRAMES", "frames/sessions")))
MD3_REV = "5112966d1a723413b1c9a1e8bea272b72e647b35"
STARMIE_REV = "35192e10a54e36eabe0a7cc57a2c1aab371cafc5"
MD2_REV = "2025-06-21"

# Offline, and every cache the remote code or torch.compile touches stays in this folder.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_MODULES_CACHE"] = f"{HERE}/modules"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"{HERE}/inductor"
os.environ["TRITON_CACHE_DIR"] = f"{HERE}/triton"

import torch  # noqa: E402
from PIL import Image  # noqa: E402

VQA_FRAMES = ["q01", "q08", "q15", "q36", "q47"]
VQA_QUESTION = "What objects are in this image?"


def gtt_mb() -> int:
    dev = os.path.dirname(glob.glob("/sys/class/drm/card*/device/mem_info_gtt_used")[0])
    return int(open(f"{dev}/mem_info_gtt_used").read()) // 2**20


def noun(query: str) -> str:
    return re.sub(r"^(the|a|an)\s+", "", query.strip(), flags=re.I)


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    r = fn()
    torch.cuda.synchronize()
    return r, (time.perf_counter() - t0) * 1e3


def load(which: str):
    from transformers import AutoModelForCausalLM

    if which == "md2":
        m = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2", revision=MD2_REV, trust_remote_code=True,
            local_files_only=True, dtype=torch.bfloat16, device_map={"": "cuda"})
        return m.eval()
    # The remote code calls Tokenizer.from_pretrained("moondream/starmie-v1") (a network fetch
    # of main). Serve it the pinned local tokenizer.json instead.
    import tokenizers

    tok = glob.glob(f"{HERE}/hf/models--moondream--starmie-v1/snapshots/{STARMIE_REV}/tokenizer.json")[0]
    tokenizers.Tokenizer.from_pretrained = staticmethod(lambda *a, **k: tokenizers.Tokenizer.from_file(tok))
    m = AutoModelForCausalLM.from_pretrained(
        "moondream/moondream3-preview", revision=MD3_REV, cache_dir=f"{HERE}/hf",
        trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": "cuda"})
    return m.eval()


def main(which: str, compile_: bool) -> None:
    tag = which + ("_compiled" if compile_ else "")
    queue = json.load(open(f"{DATA}/grounding_queries.json"))["items"]
    summary = dict(tag=tag, torch=torch.__version__)
    import transformers

    summary["transformers"] = transformers.__version__
    gtt0 = gtt_mb()
    t0 = time.perf_counter()
    model = load(which)
    summary["load_s"] = round(time.perf_counter() - t0, 1)
    if compile_:
        t0 = time.perf_counter()
        # HfMoondream has no compile(); nn.Module.compile would only wrap forward (unused).
        model._setup_caches()
        model.model.compile()
        summary["compile_s"] = round(time.perf_counter() - t0, 1)
    summary["gtt_after_load_mb"] = gtt_mb() - gtt0
    print("loaded", summary, flush=True)

    first = Image.open(f"{DATA}/{queue[0]['frame']}").convert("RGB")
    t0 = time.perf_counter()
    for _ in range(2):  # kernel selection / compile warm-up off the clock
        model.detect(first, "monitor")
        model.point(first, "monitor")
    summary["warmup_s"] = round(time.perf_counter() - t0, 1)
    print("warm", summary["warmup_s"], flush=True)

    det = open(f"{HERE}/results/{tag}_detect.jsonl", "w")
    pt = open(f"{HERE}/results/{tag}_point.jsonl", "w")
    enc_ms = []
    with torch.inference_mode():
        for q in queue:
            img = Image.open(f"{DATA}/{q['frame']}").convert("RGB")
            obj = noun(q["object"])
            r, ms = timed(lambda: model.detect(img, obj))
            boxes = [[o["x_min"], o["y_min"], o["x_max"], o["y_max"]] for o in r["objects"]]
            r, ms2 = timed(lambda: model.point(img, obj))
            pts = [[p["x"], p["y"]] for p in r["points"]]
            enc, e_ms = timed(lambda: model.encode_image(img))
            _, d_ms = timed(lambda: model.detect(enc, obj))
            enc_ms.append((e_ms, d_ms))
            det.write(json.dumps(dict(id=q["id"], query=obj, boxes=boxes, ms=round(ms),
                                      enc_ms=round(e_ms), detect_on_enc_ms=round(d_ms))) + "\n")
            pt.write(json.dumps(dict(id=q["id"], query=obj, points=pts, ms=round(ms2))) + "\n")
            det.flush()
            pt.flush()
            print(q["id"], obj, round(ms), boxes[:2], round(ms2), pts[:2],
                  f"enc {e_ms:.0f} det-on-enc {d_ms:.0f}", flush=True)
    summary["peak_alloc_mb"] = torch.cuda.max_memory_allocated() // 2**20
    summary["gtt_peak_mb"] = gtt_mb() - gtt0

    # caption/query latency only (no accuracy claims); greedy where the API allows it
    vqa = open(f"{HERE}/results/{tag}_vqa.jsonl", "w")
    by_id = {q["id"]: q for q in queue}
    with torch.inference_mode():
        for qid in VQA_FRAMES:
            img = Image.open(f"{DATA}/{by_id[qid]['frame']}").convert("RGB")
            row = dict(id=qid)
            r, row["caption_short_ms"] = timed(lambda: model.caption(img, length="short"))
            row["caption_short"] = r["caption"]
            r, row["caption_normal_ms"] = timed(lambda: model.caption(img, length="normal"))
            row["caption_normal"] = r["caption"]
            if which == "md3":
                r, row["query_ms"] = timed(lambda: model.query(img, VQA_QUESTION, reasoning=False))
                row["query"] = r["answer"]
                r, row["query_reasoning_ms"] = timed(lambda: model.query(img, VQA_QUESTION))
                row["query_reasoning"] = r["answer"]
                row["reasoning"] = r["reasoning"]["text"]
            else:
                r, row["query_ms"] = timed(lambda: model.query(img, VQA_QUESTION))
                row["query"] = r["answer"]
            for k in list(row):
                if k.endswith("_ms"):
                    row[k] = round(row[k])
            vqa.write(json.dumps(row) + "\n")
            vqa.flush()
            print("vqa", {k: v for k, v in row.items() if k.endswith("ms")}, flush=True)
    json.dump(summary, open(f"{HERE}/results/{tag}_summary.json", "w"), indent=1)
    print("summary", summary)
    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["md2", "md3"])
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()
    main(a.which, a.compile)
