# times the host patch embedding as step8.put_x0 runs it: the real-weights lin_proj (bf16) on a 256x256x3 frame
import sys, time, types
import numpy as np, torch
tn = types.ModuleType("tn")
exec(open("/tapnext/model.py").read().replace("-> TAPNext:", "-> \"TAPNext\":"), tn.__dict__)
m = tn.TAPNext(); tn.load_state(m, "/tapnext/tapnextpp_256.safetensors"); m.to_half(torch.bfloat16).eval()
conv = m.lin_proj
print("weight", tuple(conv.weight.shape), conv.weight.dtype, "abs mean", float(conv.weight.float().abs().mean()), "stride", conv.stride)
frame = torch.rand(256, 256, 3) * 2 - 1
x = frame.permute(2, 0, 1)[None].to(torch.bfloat16)
print("default torch threads in this process:", torch.get_num_threads())
for th in (1, 2, 4, 8, 16, 32):
    torch.set_num_threads(th)
    with torch.no_grad():
        for _ in range(10): y = conv(x)
        ts = []
        for _ in range(200):
            t0 = time.perf_counter(); y = conv(x); ts.append(time.perf_counter() - t0)
        # the whole put_x0 conv line incl. permute/convert/flatten/pos-emb add
        tw = []
        for _ in range(200):
            t0 = time.perf_counter()
            z = conv(frame.permute(2, 0, 1)[None].to(torch.bfloat16))[0].flatten(1).transpose(0, 1) + m.image_pos_emb[0]
            tw.append(time.perf_counter() - t0)
    med = np.median(ts)
    print(f"threads {th:2d}: conv median {med*1e3:.3f} ms (min {min(ts)*1e3:.3f}) = {302e6/med/1e9:.0f} GFLOP/s; with permute/convert/+pos median {np.median(tw)*1e3:.3f} ms; out {tuple(y.shape)} {y.dtype}")
