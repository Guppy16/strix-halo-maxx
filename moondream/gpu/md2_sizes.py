import os, time, statistics, torch
from PIL import Image
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("vikhyatk/moondream2", revision="2025-06-21", trust_remote_code=True, torch_dtype=torch.float16, device_map={"": "cuda"}).eval()
print("param dtypes:", {p.dtype for p in model.parameters()}, flush=True)
cfg=str(getattr(model,"config",""))
import re; print("config crop/vision bits:", re.findall(r"[^,{]*(?:crop|max_crops|enc_n|proj_inner|tile)[^,}]*", cfg)[:12], flush=True)
base = Image.open("/tmp/md2/judged_0060_448.jpg").convert("RGB")
Q="Is there a glass mug in this image? Answer yes or no."
def bench(im, label):
    enc=[];qry=[]
    for i in range(6):
        torch.cuda.synchronize(); t0=time.time(); e=model.encode_image(im); torch.cuda.synchronize(); t1=time.time()
        a=model.query(e,Q)["answer"]; torch.cuda.synchronize(); t2=time.time()
        if i: enc.append(t1-t0); qry.append(t2-t1)
    print(f"{label} size={im.size}: encode {statistics.median(enc)*1000:.0f} ms, query {statistics.median(qry)*1000:.0f} ms, total {(statistics.median(enc)+statistics.median(qry))*1000:.0f} ms -> {a!r}", flush=True)
bench(base, "448w")
for w in (378, 320, 256):
    bench(base.resize((w, int(base.height*w/base.width))), f"{w}w")
sq = base.copy(); sq.thumbnail((378,378)); bench(sq, "fit378")
sq2 = base.resize((378,378)); bench(sq2, "squash378x378")
bench(Image.open(os.path.join(os.environ.get("MD_FRAMES", "frames"), "sessA/judged_0060.jpg")).convert("RGB"), "orig720")
try:
    model.half(); print("param dtypes after half():", {p.dtype for p in model.parameters()}, flush=True); bench(base, "448w-fp16"); bench(sq2, "squash378-fp16")
except Exception as ex: print("fp16 failed:", repr(ex)[:300])
print(f"peak vram {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
