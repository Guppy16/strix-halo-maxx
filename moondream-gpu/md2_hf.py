import time, statistics, sys, os, torch
from PIL import Image
from transformers import AutoModelForCausalLM
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
import transformers; print("transformers", transformers.__version__, flush=True)
rev = os.environ.get("MD2_REV", "2025-06-21")
t=time.time()
model = AutoModelForCausalLM.from_pretrained("vikhyatk/moondream2", revision=rev, trust_remote_code=True, dtype=torch.float16, device_map={"": "cuda"})
model.eval()
print(f"loaded rev={rev} in {time.time()-t:.1f}s; dtype={next(model.parameters()).dtype}; vram alloc={torch.cuda.memory_allocated()/2**20:.0f} MiB", flush=True)
imgs={n: Image.open(f"/tmp/md2/{n}_448.jpg").convert("RGB") for n in ("judged_0060","judged_0000")}
qs=["Is there a glass mug in this image? Answer yes or no.","Is there a cat in this image? Answer yes or no.","Is there a table in this image? Answer yes or no."]
for n,im in imgs.items():
    for q in qs:
        torch.cuda.synchronize(); t=time.time()
        a=model.query(im, q, settings={"max_tokens": 4, "temperature": 0.0} if False else None)["answer"]
        torch.cuda.synchronize(); print(f"{n} | {q} -> {a!r} ({(time.time()-t)*1000:.0f} ms)", flush=True)
# split timing: encode vs query, median of 5 after warm-up
for n,im in imgs.items():
    enc=[]; qry=[]; tot=[]
    for i in range(6):
        torch.cuda.synchronize(); t0=time.time()
        e=model.encode_image(im)
        torch.cuda.synchronize(); t1=time.time()
        a=model.query(e, qs[0])["answer"]
        torch.cuda.synchronize(); t2=time.time()
        if i: enc.append(t1-t0); qry.append(t2-t1); tot.append(t2-t0)
    print(f"{n}: encode_image median {statistics.median(enc)*1000:.0f} ms, query(after encode) median {statistics.median(qry)*1000:.0f} ms, total median {statistics.median(tot)*1000:.0f} ms (n=5) answer={a!r}", flush=True)
print(f"peak vram {torch.cuda.max_memory_allocated()/2**20:.0f} MiB", flush=True)
