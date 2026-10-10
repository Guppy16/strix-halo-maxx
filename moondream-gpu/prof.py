import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from torch.profiler import profile, ProfilerActivity
model = load_model(); mm = model.model
im = prep(f"{SESS}/sessA/judged_0060.jpg")
Q = question("mug")
for _ in range(3):
    e = model.encode_image(im); model.query(e, Q, settings={"temperature": 0, "max_tokens": 4})
torch.cuda.synchronize()
for name, fn in [("encode", lambda: model.encode_image(im)), ("query", lambda: model.query(e, Q, settings={"temperature": 0, "max_tokens": 4}))]:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn(); torch.cuda.synchronize()
    ka = prof.key_averages()
    nk = sum(ev.count for ev in ka if ev.device_type.name == "CUDA" or (ev.self_device_time_total > 0))
    print(f"==== {name}: kernel events ~{nk}")
    print(ka.table(sort_by="self_device_time_total", row_limit=25, max_name_column_width=70))
