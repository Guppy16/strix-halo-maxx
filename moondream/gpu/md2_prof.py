import time, torch
from PIL import Image
from transformers import AutoModelForCausalLM
from torch.profiler import profile, ProfilerActivity
model = AutoModelForCausalLM.from_pretrained("vikhyatk/moondream2", revision="2025-06-21", trust_remote_code=True, device_map={"": "cuda"}).eval()
im = Image.open("/tmp/md2/judged_0060_448.jpg").convert("RGB").resize((378,378))
for _ in range(3): model.encode_image(im)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    e = model.encode_image(im); torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=14, max_name_column_width=60))
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=8, max_name_column_width=60))
# sdpa backend check
print("sdpa flash/mem-efficient enabled:", torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled(), torch.backends.cuda.math_sdp_enabled())
