"""Round 11: where the host part of Step8.step goes (everything but the NPU's run.wait).

"step" excludes the duplicate conv + queries inside x0_rows ("rows"); "write" = packing + bo write + sync.
Times each piece of step() separately over many steps, for several torch thread counts, and records the machine's
load (other tenants' CPU use) next to the numbers.
    python3 r11/hostprof.py --build full12r10r [--iters 40] [--threads 8,4,2,1]
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/work/npu-tapnext")
import step8 as S8  # noqa: E402
from step8 import P, Step8, demo_queries, preprocess  # noqa: E402


def loadavg():
    return open("/proc/loadavg").read().split()[0]


def cpu_busy():
    """Fraction of all CPUs busy over the last 0.2 s (/proc/stat), i.e. what other work takes."""
    def snap():
        v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
        return sum(v), v[3] + v[4]
    t0, i0 = snap()
    time.sleep(0.2)
    t1, i1 = snap()
    return 1 - (i1 - i0) / max(t1 - t0, 1)


def profile(s, video, q, iters, warmup):
    pyxrt = S8.pyxrt
    parts = {k: [] for k in ("conv", "queries", "rows", "write", "launch", "wait", "read", "heads", "step_total", "dispatch_us")}
    m = s.model
    for it in range(warmup + iters):
        rel = torch.cat([q[..., :1] - it, q[..., 1:]], -1)
        frame = video[it % 2]
        T = [time.perf_counter()]
        with torch.no_grad():
            x = m.lin_proj(frame.permute(2, 0, 1)[None].to(torch.bfloat16))[0].flatten(1).transpose(0, 1) + m.image_pos_emb[0]
            T.append(time.perf_counter())
            m.embed_queries(rel)
            T.append(time.perf_counter())
        rows = s.x0_rows(frame, rel)  # conv + queries again (timed above); the packing below is what "q16" measures
        T.append(time.perf_counter())
        s.write_x0(rows)
        T.append(time.perf_counter())
        r = s.kern(3, s.bo_i, s.n_instr, s.bo_w, s.bo_a)
        T.append(time.perf_counter())
        st = r.wait(3_000_000)
        assert str(st) == "ert_cmd_state.ERT_CMD_STATE_COMPLETED", st
        T.append(time.perf_counter())
        if s.first:
            s.set_first(False)
        xq = s.read_col(s.info["final"])[16:16 + q.shape[1]]
        T.append(time.perf_counter())
        s.heads(xq)
        T.append(time.perf_counter())
        if it >= warmup:
            d = np.diff(T) * 1e3
            for k, v in zip(("conv", "queries", "rows", "write", "launch", "wait", "read", "heads"), d):
                parts[k].append(v)
            parts["step_total"].append((T[-1] - T[0]) * 1e3 - d[2] + 0.0)
            parts["dispatch_us"].append((T[6] - T[4]) * 1e3)
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="full12r10r")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--threads", default="8,4,2,1")
    o = ap.parse_args()
    frames = np.load("/work/tapnet/data/clip30.npy")[:2]
    video = preprocess(frames)
    s = Step8(o.build, o.depth)
    q = demo_queries()
    s.reset()
    print(f"HOST_FAST={S8.HOST_FAST} HEAD_THREADS={S8.HEAD_THREADS} load {loadavg()} cpu busy before {cpu_busy():.3f} (all {os.cpu_count()} cpus)")
    for th in [int(t) for t in o.threads.split(",")]:
        torch.set_num_threads(th)
        parts = profile(s, video, q, o.iters, o.warmup)
        host = np.array(parts["step_total"]) - np.array(parts["wait"]) - np.array(parts["launch"])
        print(f"threads {th}: step median {np.median(parts['step_total']):.2f} ms, host (step - launch - wait) median "
              f"{np.median(host):.2f} p90 {np.percentile(host, 90):.2f}; load {loadavg()}")
        for k in ("conv", "queries", "write", "launch", "wait", "read", "heads"):
            v = np.array(parts[k])
            print(f"   {k:8s} median {np.median(v):7.3f}  p10 {np.percentile(v, 10):7.3f}  p90 {np.percentile(v, 90):7.3f} ms")


if __name__ == "__main__":
    main()
