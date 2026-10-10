import time, statistics, torch, os
from PIL import Image
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("vikhyatk/moondream2", revision="2025-06-21", trust_remote_code=True, device_map={"": "cuda"}).eval()
im = Image.open("/tmp/md2/judged_0060_448.jpg").convert("RGB").resize((378,378))
Q="Is there a glass mug in this image? Answer yes or no."
enc=[];qry=[]
t=time.time()
for i in range(8):
    torch.cuda.synchronize(); t0=time.time(); e=model.encode_image(im); torch.cuda.synchronize(); t1=time.time()
    a=model.query(e,Q)["answer"]; torch.cuda.synchronize(); t2=time.time()
    if i>2: enc.append(t1-t0); qry.append(t2-t1)
print(f"[{os.environ.get("LABEL")}] 378x378: encode {statistics.median(enc)*1000:.0f} ms, query {statistics.median(qry)*1000:.0f} ms, total {(statistics.median(enc)+statistics.median(qry))*1000:.0f} ms -> {a!r} (loop wall {time.time()-t:.0f}s)", flush=True)
