# Fetches only the safetensors weights (modelv2-*, the shards model.safetensors.index.json maps to)
# and the starmie-v1 tokenizer, both pinned. model_fp8.pt (pickle) and the unused model-* shards are skipped.
import huggingface_hub as h

MD3 = "5112966d1a723413b1c9a1e8bea272b72e647b35"
STARMIE = "35192e10a54e36eabe0a7cc57a2c1aab371cafc5"
print(h.snapshot_download("moondream/moondream3-preview", revision=MD3, cache_dir="hf",
                          allow_patterns=["modelv2-*.safetensors"]))
print(h.snapshot_download("moondream/starmie-v1", revision=STARMIE, cache_dir="hf",
                          allow_patterns=["*.json", "*.md", ".gitattributes"]))
print("DOWNLOAD_DONE")
