import sys, os, torch, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import md2fast
from md2fast import FastMD2
from torch.profiler import profile, ProfilerActivity
ap = argparse.ArgumentParser(); ap.add_argument("--dtype", default="bf16"); ap.add_argument("--lin", default="Lin"); ap.add_argument("--rows", type=int, default=20); ap.add_argument("--tlin", default=""); ap.add_argument("--pool", type=int, default=0); ap.add_argument("--graph", action="store_true"); ap.add_argument("--tunable", action="store_true"); ap.add_argument("--nofuse", action="store_true")
args = ap.parse_args()
dt = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
if args.tunable:
    torch.cuda.tunable.enable(True); torch.cuda.tunable.tuning_enable(True)
    torch.cuda.tunable.set_filename(os.path.expanduser(f"~/moondream-opt/tunable_{args.dtype}_default.csv"))
import tgemm
def _cls(n): return getattr(md2fast, n) if hasattr(md2fast, n) else getattr(tgemm, n)
model = load_model(); fast = FastMD2(model, dtype=dt, lin_cls=_cls(args.lin), tlin_cls=_cls(args.tlin) if args.tlin else None, pool=args.pool, fuse=not args.nofuse)
im = prep(f"{SESS}/sessA/judged_0060.jpg"); fast.set_image(im); fast.set_question(question("mug"))
for _ in range(3): fast.encode(); fast.query()
if args.tunable: torch.cuda.tunable.tuning_enable(False)
if args.graph: fast.capture()
torch.cuda.synchronize()
for name, fn in [("encode", fast.encode), ("query", fast.query)]:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn(); torch.cuda.synchronize()
    ka = prof.key_averages()
    print(f"==== {name}")
    print(ka.table(sort_by="self_device_time_total", row_limit=args.rows, max_name_column_width=80))
