# Re-download after deletion (2026-10-09, round-2 VQA): the same two passes as NOTES.md, pinned.
# Pass 1: code/config/docs only. Pass 2: the four modelv2-* safetensors shards + starmie-v1 json/md.
# Never model_fp8.pt, never the old model-* shards, no PNGs.
import huggingface_hub as h

MD3 = "5112966d1a723413b1c9a1e8bea272b72e647b35"
STARMIE = "35192e10a54e36eabe0a7cc57a2c1aab371cafc5"
print(h.snapshot_download("moondream/moondream3-preview", revision=MD3, cache_dir="hf",
                          allow_patterns=["*.py", "*.json", "*.md", ".gitattributes"]))
print(h.snapshot_download("moondream/moondream3-preview", revision=MD3, cache_dir="hf",
                          allow_patterns=[f"modelv2-0000{i}-of-00004.safetensors" for i in range(1, 5)]))
print(h.snapshot_download("moondream/starmie-v1", revision=STARMIE, cache_dir="hf",
                          allow_patterns=["*.json", "*.md", ".gitattributes"]))
print("DOWNLOAD_DONE")
