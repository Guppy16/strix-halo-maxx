"""design8 test driver. `sched NAME` writes build/NAME.sched (+ .meta) from the test's plan (pure python);
`run NAME [--iters N --warmup N --timeout-ms N]` packs the test's data, dispatches and checks (needs the NPU).

    python3 run8.py sched t1 ; b.sh t1 --sched /work/npu-tapnext/build/t1.sched ; python3 run8.py run t1
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from plan8 import COLS, M, OBJ, PAN, PATCH_PARAM_BYTES, QKV_PER_H, QKV_PER_HC, RB, S, SSM_PARAM_BYTES, Plan, SLOT, UNIT, XB

HERE = Path(__file__).resolve().parent
M_REAL = int(os.environ.get("M_REAL", "1072"))


# ---------------------------------------------------------------- test plans ---------------------------------
def plan_t1():
    """GEMM -> Q16, rms norm with a residual, GEMM from its panels, LN norm, GELU GEMM."""
    p = Plan()
    x0 = p.q16("X0")
    wp = p.alloc_w(16 * SLOT)
    g1 = p.gemm("g1", x0, 768, 4)
    n1 = p.norm("n1", "rms", [g1, x0], wp)
    g2 = p.gemm("g2", n1, 768, 4)
    n2 = p.norm("n2", "ln", [g2, n1], wp + 3 * SLOT)
    g3 = p.gemm("g3", n2, 768, 4, epi="gelu")
    return p, dict(pre_n=0, body_n=len(p.ops), reps=1, x0=x0, params_w=wp)


def plan_t2():
    """gated GELU pairs -> K=1536 GEMMs (bias on one), 3-input LN, gelu -> panels -> K=1536 GEMM."""
    p = Plan()
    d1, d2, x0 = p.q16("D1"), p.q16("D2"), p.q16("X0")  # the 3-input norm reads [D1, D2, X0] at equal distances
    wp = p.alloc_w(16 * SLOT)
    g = p.apan("G", nq=48)
    p.gemm("up_a", x0, 768, 16, epi="gated", dst=dict(g, q0=0))
    p.gemm("up_b", x0, 768, 16, epi="gated", dst=dict(g, q0=24))
    p.gemm("down_a", dict(g, q0=0), 1536, 4, dst=d1)
    p.gemm("down_b", dict(g, q0=24), 1536, 4, dst=d2, bias_in=False)
    r2 = p.norm("n2", "ln", [d1, d2, x0], wp)
    m = p.apan("M", nq=48)
    p.gemm("mlp0", r2, 768, 16, epi="gelu_pan", dst=dict(m, q0=0))
    p.gemm("mlp3_a", dict(m, q0=0), 1536, 4)
    p.gemm("mlp3_b", dict(m, q0=24), 1536, 4, bias_in=False)
    return p, dict(pre_n=0, body_n=len(p.ops), reps=1, x0=x0, params_w=wp)


def plan_t1a():
    p = Plan()
    x0 = p.q16("X0")
    wp = p.alloc_w(16 * SLOT)
    g1 = p.gemm("g1", x0, 768, 4)
    p.norm("n1", "rms", [g1, x0], wp)
    return p, dict(pre_n=0, body_n=len(p.ops), reps=1, x0=x0, params_w=wp)


def plan_t1b():
    p = Plan()
    x0 = p.q16("X0")
    wp = p.alloc_w(16 * SLOT)
    g1 = p.gemm("g1", x0, 768, 4)
    n1 = p.norm("n1", "rms", [g1, x0], wp)
    p.gemm("g2", n1, 768, 4)
    return p, dict(pre_n=0, body_n=len(p.ops), reps=1, x0=x0, params_w=wp)


def plan_t3():
    """SSM op on a block's area, then a GEMM from its z panels."""
    p = Plan()
    area = p.ssm_area("A0")
    wp = p.alloc_w(12 * SSM_PARAM_BYTES)
    p.ssm("ssm", area, wp)
    p.gemm("g", area["z"], 768, 4)
    return p, dict(pre_n=0, body_n=len(p.ops), reps=1, x0=None, params_w=wp, area=area)


def plan_t4():
    """qkv GEMM with the attention-object epilogue, attention on the GEMM cores, out_proj from the O panels."""
    p = Plan()
    x0 = p.q16("X0")
    wp = p.alloc_w(16 * SLOT)
    qkv = p.gemm("qkv", x0, 768, 12, epi="qkv")
    o = p.attn("attn", qkv)
    p.gemm("out_proj", o, 768, 4)
    return p, dict(pre_n=0, body_n=len(p.ops), reps=1, x0=x0, params_w=wp)


def plan_t4q():
    """the qkv GEMM (attention-object epilogue, 12 drains per column) followed by a plain GEMM; no attention."""
    p = Plan()
    x0 = p.q16("X0")
    wp = p.alloc_w(16 * SLOT)
    p.gemm("qkv", x0, 768, 12, epi="qkv")
    p.gemm("g2", x0, 768, 4)
    return p, dict(pre_n=0, body_n=len(p.ops), reps=1, x0=x0, params_w=wp)


def plan_t5():
    """patch embedding (EW op from host bf16 patches) + pos-emb add / rms norm + a GEMM from its panels."""
    p = Plan()
    frame = p.alloc(COLS * RB * 3072, "FRAME")
    pe = p.q16("PE")
    pos = p.q16("POS")
    wpatch = p.alloc_w(12 * PATCH_PARAM_BYTES)
    wp = p.alloc_w(16 * SLOT)
    p.patch("patch", frame, wpatch, dst=pe)
    r0 = p.norm("n0", "rms", [pe, pos], wp)
    p.gemm("g", r0, 768, 4)
    return p, dict(pre_n=0, body_n=len(p.ops), reps=1, x0=None, params_w=wp, frame=frame, pe=pe, pos=pos, wpatch=wpatch)


def plan_t2a():
    """gated GELU pairs only (tiles-per-object 2)."""
    p = Plan()
    x0 = p.q16("X0")
    wp = p.alloc_w(16 * SLOT)
    g = p.apan("G", nq=48)
    p.gemm("up_a", x0, 768, 16, epi="gated", dst=dict(g, q0=0))
    p.gemm("up_b", x0, 768, 16, epi="gated", dst=dict(g, q0=24))
    return p, dict(pre_n=0, body_n=len(p.ops), reps=1, x0=x0, params_w=wp)


def plan_t2b():
    """gelu -> panels, then the two K=1536 GEMMs."""
    p = Plan()
    x0 = p.q16("X0")
    wp = p.alloc_w(16 * SLOT)
    m = p.apan("M", nq=48)
    p.gemm("mlp0", x0, 768, 16, epi="gelu_pan", dst=dict(m, q0=0))
    p.gemm("mlp3_a", dict(m, q0=0), 1536, 4)
    p.gemm("mlp3_b", dict(m, q0=24), 1536, 4, bias_in=False)
    return p, dict(pre_n=0, body_n=len(p.ops), reps=1, x0=x0, params_w=wp)


def plan_t2c():
    """gated pairs then the two K=1536 consumers (dependencies, no EW op)."""
    p = Plan()
    x0 = p.q16("X0")
    wp = p.alloc_w(16 * SLOT)
    g = p.apan("G", nq=48)
    p.gemm("up_a", x0, 768, 16, epi="gated", dst=dict(g, q0=0))
    p.gemm("up_b", x0, 768, 16, epi="gated", dst=dict(g, q0=24))
    p.gemm("down_a", dict(g, q0=0), 1536, 4)
    p.gemm("down_b", dict(g, q0=24), 1536, 4, bias_in=False)
    return p, dict(pre_n=0, body_n=len(p.ops), reps=1, x0=x0, params_w=wp)


def plan_t2d():
    """three plain GEMMs then a 3-input LayerNorm over their outputs, then a GEMM from its panels."""
    p = Plan()
    x0 = p.q16("X0")
    wp = p.alloc_w(16 * SLOT)
    o1, o2, o3 = p.q16("O1"), p.q16("O2"), p.q16("O3")
    p.gemm("g1", x0, 768, 4, dst=o1)
    p.gemm("g2", x0, 768, 4, dst=o2)
    p.gemm("g3", x0, 768, 4, dst=o3)
    r2 = p.norm("n2", "ln", [o1, o2, o3], wp)
    p.gemm("g4", r2, 768, 4)
    return p, dict(pre_n=0, body_n=len(p.ops), reps=1, x0=x0, params_w=wp)


def plan_t2e():
    """t2 without the norm: gated -> K=1536, then gelu_pan -> K=1536 (8 GEMM ops)."""
    p = Plan()
    x0 = p.q16("X0")
    wp = p.alloc_w(16 * SLOT)
    g = p.apan("G", nq=48)
    p.gemm("up_a", x0, 768, 16, epi="gated", dst=dict(g, q0=0))
    p.gemm("up_b", x0, 768, 16, epi="gated", dst=dict(g, q0=24))
    p.gemm("down_a", dict(g, q0=0), 1536, 4)
    p.gemm("down_b", dict(g, q0=24), 1536, 4, bias_in=False)
    m = p.apan("M", nq=48)
    p.gemm("mlp0", x0, 768, 16, epi="gelu_pan", dst=dict(m, q0=0))
    p.gemm("mlp3_a", dict(m, q0=0), 1536, 4)
    p.gemm("mlp3_b", dict(m, q0=24), 1536, 4, bias_in=False)
    return p, dict(pre_n=0, body_n=len(p.ops), reps=1, x0=x0, params_w=wp)


def plan_t2f():
    """t2 without the mlp ops: gated -> K=1536 -> 3-input LN."""
    p = Plan()
    d1, d2, x0 = p.q16("D1"), p.q16("D2"), p.q16("X0")
    wp = p.alloc_w(16 * SLOT)
    g = p.apan("G", nq=48)
    p.gemm("up_a", x0, 768, 16, epi="gated", dst=dict(g, q0=0))
    p.gemm("up_b", x0, 768, 16, epi="gated", dst=dict(g, q0=24))
    p.gemm("down_a", dict(g, q0=0), 1536, 4, dst=d1)
    p.gemm("down_b", dict(g, q0=24), 1536, 4, dst=d2, bias_in=False)
    p.norm("n2", "ln", [d1, d2, x0], wp)
    return p, dict(pre_n=0, body_n=len(p.ops), reps=1, x0=x0, params_w=wp)


def plan_full(depth=12):
    """the whole frame (`depth` blocks): timing with random weights (no checks). Names `full<depth>...` map here."""
    from plan8 import full_step
    p, info = full_step(depth)
    return p, dict(pre_n=0, body_n=(len(p.ops) - 1) // depth, reps=depth, x0=None, params_w=None, full=info)


def plan_full12():
    return plan_full(12)


def plan_g11(reps=12):
    """the 11 GEMM ops of a block x reps (no EW ops, no attention): the GEMM frame's own time (cf. s7_block 35.1 ms)."""
    p = Plan()
    x0 = p.q16("X0")
    wp = p.alloc_w(16 * SLOT)
    g = p.apan("G", nq=48)
    m = p.apan("M", nq=48)
    for rep in range(reps):
        p.gemm(f"yx{rep}", x0, 768, 8, epi="gelu", flags=["gelu"] * 12 + ["plain"] * 12)
        p.gemm(f"out{rep}", x0, 768, 4)
        p.gemm(f"up_a{rep}", x0, 768, 16, epi="gated", dst=dict(g, q0=0))
        p.gemm(f"up_b{rep}", x0, 768, 16, epi="gated", dst=dict(g, q0=24))
        p.gemm(f"down_a{rep}", dict(g, q0=0), 1536, 4)
        p.gemm(f"down_b{rep}", dict(g, q0=24), 1536, 4, bias_in=False)
        p.gemm(f"qkv{rep}", x0, 768, 12, epi="qkv")
        p.gemm(f"out_proj{rep}", x0, 768, 4)
        p.gemm(f"mlp0{rep}", x0, 768, 16, epi="gelu_pan", dst=dict(m, q0=0))
        p.gemm(f"mlp3_a{rep}", dict(m, q0=0), 1536, 4)
        p.gemm(f"mlp3_b{rep}", dict(m, q0=24), 1536, 4, bias_in=False)
    return p, dict(pre_n=0, body_n=11, reps=reps, x0=x0, params_w=wp, full=dict(timing=True))


TESTS = dict(t3cp=plan_t3, g11=plan_g11, g11o2=plan_g11, t3ng=plan_t3, t3o2=plan_t3, t4k48=plan_t4, full12na=plan_full12, full12ns=plan_full12, t1=plan_t1, t2=plan_t2, t1a=plan_t1a, t1b=plan_t1b, t3=plan_t3, t4=plan_t4, t4q=plan_t4q, t4s=plan_t4, t4b=plan_t4, t4c=plan_t4, t5=plan_t5,
             t2a=plan_t2a, t2b=plan_t2b, t2c=plan_t2c, t2d=plan_t2d, t2e=plan_t2e, t2f=plan_t2f, full12r=plan_full12, full12=plan_full12)


def plan_of(name):
    """test plans by name; `full<depth><suffix>` (e.g. full1, full12f) is the whole step at that depth."""
    import re
    if "-" in name:  # round 6: `<test>-<variant>` (e.g. t2-r6a) builds test <test> with other flags
        name = name.split("-")[0]
    if name in TESTS:
        return TESTS[name]()
    for prefix in ("t3", "t4", "t1", "t2"):  # variants of a test (build flags differ): t3o2, t3cp, ...
        if name.startswith(prefix) and not name.startswith("t2") or name.startswith(prefix) and name[:3] in TESTS:
            return TESTS.get(name[:3], TESTS[prefix])()
    m = re.match(r"full(\d+)", name)
    assert m, name
    return plan_full(int(m.group(1)))


def write_sched(name):
    p, info = plan_of(name)
    b = HERE / "build"
    b.mkdir(exist_ok=True)
    (b / f"{name}.sched").write_text(p.sched_json())
    (b / f"{name}.sched.meta").write_text(json.dumps(dict(w_words=max(p.w_bytes, 64) // 4, act_words=max(p.act_bytes, 64) // 4,
                                                            pre_n=info["pre_n"], body_n=info["body_n"], reps=info["reps"])))
    print(f"{name}: {len(p.ops)} ops, ACT {p.act_bytes / 1e6:.1f} MB, W {p.w_bytes / 1e6:.1f} MB")


# ---------------------------------------------------------------- running ---------------------------------------
def run(name, o):
    import numpy as np
    import pyxrt
    from ml_dtypes import bfloat16

    import pack8 as P
    from pack import decode_rows

    p, info = plan_of(name)
    meta = json.loads((HERE / "build" / f"{name}.sched.meta").read_text())
    rng = np.random.default_rng(0)
    W = np.zeros(meta["w_words"], np.uint32)
    ACT = np.zeros(meta["act_words"], np.uint32)
    f32 = lambda v: np.asarray(v, np.float32)
    bfv = lambda v: f32(v).astype(bfloat16)

    def put(region, words):
        ACT[region["base"] // 4: region["base"] // 4 + words.size] = words

    def get(region):
        n = COLS * region["col"] // 4
        return ACT_out[region["base"] // 4: region["base"] // 4 + n]

    def get_bytes(base, nbytes):
        return ACT_out[base // 4: (base + nbytes) // 4].view(np.uint8)

    # host inputs: X0 = random bf16 rows (X' parts) and their device-style panels
    x0 = np.zeros((COLS * M, 768), bfloat16)
    x0[:M_REAL] = bfv(rng.standard_normal((M_REAL, 768)) * 1.5 + 0.2)
    if info.get("x0") is not None:
        put(info["x0"], P.to_q16(x0) | P.panels_to_q16_words(P.encode_hw(f32(x0))))
    patch = None
    if info.get("frame") is not None:  # the patch test: a random uint8 frame, folded conv weights, pos-emb/query tokens
        img = rng.integers(0, 256, (256, 256, 3)).astype(np.uint8)
        conv_w = (rng.standard_normal((768, 3, 8, 8)) * 0.1).astype(np.float32)
        conv_b = (rng.standard_normal(768) * 0.3).astype(np.float32)
        wf, bfold = P.conv_fold(conv_w, conv_b)
        patches = np.zeros((COLS * M, 192), np.float32)
        patches[:1024] = P.frame_patches(img)
        pb = bfv(patches)
        pw = pb.reshape(COLS, RB, 8, 24, 8).transpose(0, 1, 3, 2, 4)  # [c][rb][cb][8 rows][8]
        ACT[info["frame"] // 4: info["frame"] // 4 + pw.size // 2] = np.ascontiguousarray(pw).reshape(-1).view(np.uint32)
        for q in range(12):
            words = P.patch_params(q, wf, bfold)
            off = (info["wpatch"] + q * PATCH_PARAM_BYTES) // 4
            W[off: off + words.size] = words
        pos = bfv(rng.standard_normal((COLS * M, 768)) * 0.5)
        put(info["pos"], P.to_q16(pos))
        patch = dict(pb=pb, wf=wf, bfold=bfold, pos=pos)
    ssm = None
    if info.get("area") is not None:  # the SSM test: y, x, x-1, x-2, x-3, h0 and per-head parameters
        area = info["area"]
        rnd = lambda scale, shape=(COLS * M, 768): bfv(rng.standard_normal(shape) * scale)
        y, x, xm1, xm2, xm3 = rnd(1.0), rnd(1.0), rnd(1.0), rnd(1.0), rnd(1.0)
        h0 = (rng.standard_normal((COLS * M, 768)) * 1.0).astype(np.float32)
        first = int(os.environ.get("FIRST", "0"))
        prm = dict(conv_w=bfv(rng.standard_normal((4, 768)) * 0.3), conv_b=bfv(rng.standard_normal(768) * 0.1),
                   a_param=bfv(rng.uniform(-3, 0.5, 768)), wx=bfv(rng.standard_normal((12, 64, 64)) * 0.15),
                   wa=bfv(rng.standard_normal((12, 64, 64)) * 0.15), bx=bfv(rng.standard_normal(768) * 0.3),
                   ba=bfv(rng.standard_normal(768) * 0.3))
        ssm = dict(y=y, x=x, xm1=xm1, xm2=xm2, xm3=xm3, h0=h0, first=first, **prm)

        def put_slot(k, words):  # words = a [c][S] region image -> slot k of every column's 8 S area
            wc = words.reshape(COLS, -1)
            for c in range(COLS):
                off = (area["base"] + c * 8 * S + k * S) // 4
                ACT[off: off + wc.shape[1]] = wc[c]

        put_slot(0, P.to_q16(y))
        put_slot(1, P.to_q16(x))
        put_slot(2, P.to_q16(xm1))
        put_slot(3, P.to_q16(xm2))
        put_slot(4, P.to_q16(xm3))
        put_slot(5, P.h_units(h0, 0))
        put_slot(6, P.h_units(h0, 1))
        for q in range(12):
            sl = slice(q * 64, q * 64 + 64)
            words = P.ssm_params(q, prm["conv_w"][:, sl], prm["conv_b"][sl], prm["a_param"][sl], prm["wx"][q], prm["bx"][sl],
                                 prm["wa"][q], prm["ba"][sl], first)
            off = (info["params_w"] + q * SSM_PARAM_BYTES) // 4
            W[off: off + words.size] = words
    sc1 = bfv(1 + 0.1 * rng.standard_normal(768))
    gamma, beta = bfv(1 + 0.2 * rng.standard_normal(768)), bfv(0.3 * rng.standard_normal(768))
    if info.get("full") is not None:  # random parameters for every EW op of the full step
        for op in p.ops:
            if op["kind"] == "ew" and op["fn"] == "ssm":
                for q in range(12):
                    words = P.ssm_params(q, bfv(rng.standard_normal((4, 64)) * 0.3), bfv(rng.standard_normal(64) * 0.1), bfv(rng.uniform(-3, 0.5, 64)),
                                         bfv(rng.standard_normal((64, 64)) * 0.15), bfv(rng.standard_normal(64) * 0.3),
                                         bfv(rng.standard_normal((64, 64)) * 0.15), bfv(rng.standard_normal(64) * 0.3), 1)
                    off = (op["params_w"] + q * SSM_PARAM_BYTES) // 4
                    W[off: off + words.size] = words
            elif op["kind"] == "ew" and op["fn"] == "patch":
                wf = (rng.standard_normal((768, 3, 8, 8)) * 0.001).astype(np.float32)
                for q in range(12):
                    words = P.patch_params(q, wf, np.zeros(768, np.float32))
                    off = (op["params_w"] + q * PATCH_PARAM_BYTES) // 4
                    W[off: off + words.size] = words
    for op in p.ops:  # every norm op's params at its own W offset
        if op["kind"] == "ew" and op["fn"] == "norm":
            pw = op["params_w"] // 4
            if op["mode"] in ("rms", "none"):
                W[pw: pw + 3 * SLOT // 4] = P.norm_params(op["mode"], f32(sc1) - 1, n_in=op["n_in"])
            else:
                W[pw: pw + 5 * SLOT // 4] = P.norm_params(op["mode"], gamma, beta, n_in=op["n_in"])
    # weights per gemm op
    ws = {}
    for op in p.ops:
        if op["kind"] != "gemm":
            continue
        wt = bfv(rng.standard_normal((op["N"], op["K"])) * 0.03)
        bias = bfv(rng.standard_normal(op["N"]) * 0.5)
        ws[op["name"]] = (wt, bias)
        words = P.pack_gemm(op, f32(wt), f32(bias))
        W[op["w"] // 4: op["w"] // 4 + words.size] = words

    dev = pyxrt.device(0)
    b = str(HERE / "build" / name)
    xb = pyxrt.xclbin(b + ".xclbin")
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
    instr = np.fromfile(b + ".bin", dtype=np.uint32)
    bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
    bo_w = pyxrt.bo(dev, W.nbytes, pyxrt.bo.host_only, kern.group_id(3))
    bo_a = pyxrt.bo(dev, ACT.nbytes, pyxrt.bo.host_only, kern.group_id(4))
    bo_i.write(instr.tobytes(), 0)
    bo_w.write(W.tobytes(), 0)
    bo_a.write(ACT.tobytes(), 0)
    for bo in (bo_i, bo_w, bo_a):
        bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    tk = []
    for it in range(o.warmup + o.iters):
        t1 = time.perf_counter()
        r = kern(3, bo_i, len(instr), bo_w, bo_a)
        st = r.wait(o.timeout_ms * 1000)
        t2 = time.perf_counter()
        if str(st) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
            print("RUN FAILED", st, f"after {1e6 * (t2 - t1):.0f} us (iteration {it})")
            return 1
        if it >= o.warmup:
            tk.append(t2 - t1)
    bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    ACT_out = np.frombuffer(bo_a.read(ACT.nbytes, 0), dtype=np.uint32)
    try:
        busy = int(open("/sys/class/drm/card1/device/gpu_busy_percent").read())
    except OSError:
        busy = -1
    print(f"TIME kernel_us {1e6 * np.median(tk):.0f} min {1e6 * min(tk):.0f} iters {o.iters} gpu_busy% {busy}")

    if info.get("full") is not None:
        print(f"FULL step: {len(p.ops)} ops, ACT {ACT.nbytes / 1e6:.0f} MB, W {W.nbytes / 1e6:.0f} MB, median {1e6 * np.median(tk):.0f} us over {o.iters} iters")
        return 0
    # ---- checks: walk the ops with a float reference of what each should produce --------------------------------
    rc = 0
    xprime = {}  # region base -> bf16 [rows, 768] X' content (exact expectation)
    panels = {}  # region base -> decoded float panels the consumer multiplies (from the DEVICE output)
    if info.get("x0") is not None:
        xprime[info["x0"]["base"]] = x0
        panels[info["x0"]["base"]] = decode_rows(P.encode_hw(f32(x0))).astype(np.float64)
    apan = {}  # (base, q0) -> decoded [rows, K] float of device panels
    attn_in = None

    def report(label, got, exp, ulp_of=None):
        nonlocal rc
        got = got[:M_REAL].astype(np.float64)
        exp = exp[:M_REAL].astype(np.float64)
        mis = np.mean(got != exp)
        err = np.abs(got - exp)
        ref = np.abs(exp) if ulp_of is None else np.abs(ulp_of[:M_REAL])
        ulp = np.max(err / np.maximum(ref * 2 ** -8, 1e-30))
        print(f"CHECK {label}: mismatch {mis:.4%} max_err/ulp {ulp:.2f} rms_rel {np.sqrt((err ** 2).sum() / max((exp ** 2).sum(), 1e-30)):.2e}")
        return mis, ulp

    for op in p.ops:
        if op["kind"] == "gemm":
            wt, bias = ws[op["name"]]
            src = op["src"]
            if src["kind"] == "q16":
                a = panels.get(src["base"])
            else:
                a = apan.get((src["base"], src.get("q0", 0)))
            if a is None:
                print(f"SKIP {op['name']}: no reference A"); continue
            w_dec = decode_rows(P.encode_rows(f32(wt))).astype(np.float64)
            acc = a[:, :op["K"]] @ w_dec.T + (f32(bias) if op["bias_in"] else 0.0)
            if op["epi"] in ("plain", "gelu"):
                y = acc.astype(np.float32).astype(bfloat16)
                if op["epi"] == "gelu":
                    yg = y.astype(np.float64)
                    yg = P.gelu_ref(yg)
                    # per-tile flags: only tiles flagged gelu
                    tiles = np.array([f == "gelu" for f in op["flags"]])
                    mask = np.repeat(tiles, 64)[None, :]
                    y = np.where(mask, yg.astype(np.float32).astype(bfloat16), y)
                got = P.from_q16(get(op["dst"]), ncols=op["N"])
                mis, ulp = report(f"{op['name']} K={op['K']} N={op['N']} {op['epi']}", got, y)
                rc |= bool(mis > (0.02 if op["epi"] == "gelu" else 1e-3))
                xprime[op["dst"]["base"]] = got
            elif op["epi"] in ("gated", "gelu_pan"):
                dst = op["dst"]
                nobj = 3 * op["nt"] // (2 if op["epi"] == "gated" else 1)
                got_u8 = P.panels_from_apan(get(dst), dst, dst["nq"])[:, dst.get("q0", 0) * 8: (dst.get("q0", 0) + nobj) * 8]
                got = decode_rows(np.ascontiguousarray(got_u8)).astype(np.float64)
                if op["epi"] == "gated":
                    y = acc.astype(np.float32).astype(bfloat16).astype(np.float64).reshape(-1, op["N"] // 64, 2, 32)
                    g = P.gelu_ref(y[:, :, 0]).astype(np.float32).astype(bfloat16).astype(np.float64)
                    z = (g * y[:, :, 1]).astype(np.float32).astype(bfloat16).astype(np.float64).reshape(COLS * M, -1)
                else:  # gelu_pan (the ViT MLP): erf form
                    z = P.gelu_erf_ref(acc.astype(np.float32).astype(bfloat16).astype(np.float64)).astype(np.float32).astype(bfloat16).astype(np.float64)
                exp = decode_rows(P.encode_hw(z.astype(np.float32))).astype(np.float64)
                if os.environ.get("DBG"):
                    exp_u8 = P.encode_hw(z.astype(np.float32))
                    q0 = dst.get("q0", 0)
                    raw = get(dst).view(np.uint8).reshape(COLS, dst["nq"], dst["kstride"])
                    for (c, q, rb) in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 8, 0), (0, 9, 0), (1, 0, 0), (0, 0, 17)]:
                        gp = raw[c, q0 + q, rb * 576: rb * 576 + 576].reshape(8, 72)
                        rows = slice(c * M + rb * 8, c * M + rb * 8 + 8)
                        ep = exp_u8[rows, q * 8: q * 8 + 8].transpose(1, 0, 2).reshape(8, 72)  # [panel cb][row][9]
                        print(f"  c{c} q{q} rb{rb}: got exps {gp[:, 0].tolist()} rows0 {gp[0, :9].tolist()}")
                        print(f"             exp exps {ep[:, 0].tolist()} rows0 {ep[0, :9].tolist()}")
                        same = [int((gp[i] == ep[i]).all()) for i in range(8)]
                        print(f"             panel equal {same}")
                        if (c, q, rb) == (0, 0, 0):
                            for pi in (0, 3, 4, 7):
                                print(f"             panel {pi} got {gp[pi].tolist()}")
                                print(f"             panel {pi} exp {ep[pi].tolist()}")
                mis, ulp = report(f"{op['name']} {op['epi']} panels", got, exp, ulp_of=np.abs(exp).max(axis=1, keepdims=True) * np.ones_like(exp))
                rc |= bool(mis > 0.05)
                # register the device panels as the consumers' A (by k-step window)
                full = decode_rows(np.ascontiguousarray(P.panels_from_apan(get(dst), dst, dst["nq"]))).astype(np.float64)
                for qq in range(0, dst["nq"], 24):
                    apan[(dst["base"], qq)] = full[:, qq * 64: qq * 64 + 1536]
            elif op["epi"] == "qkv":
                y = acc.astype(np.float32).astype(bfloat16).astype(np.float32)  # [rows, 2304]
                raw = get_bytes(op["dst"]["base"], 12 * QKV_PER_H)
                qkv_dec = {}
                for kind, name in enumerate("qkv"):
                    exp_u8 = P.encode_hw(y[:, kind * 768:(kind + 1) * 768])  # [rows, 96, 9]
                    got_rows = np.zeros_like(exp_u8)
                    for h in range(12):
                        for c in range(COLS):
                            for i in range(3):
                                obj = raw[h * QKV_PER_H + c * QKV_PER_HC + (3 * i + kind) * OBJ:][:OBJ]
                                rows = slice(c * M + 48 * i, c * M + 48 * i + 48)
                                if kind < 2:
                                    pan = obj[:3456].reshape(6, 8, 8, 9)  # [rb][cb][row][9]
                                    got_rows[rows, h * 8:(h + 1) * 8] = pan.transpose(0, 2, 1, 3).reshape(48, 8, 9)
                                else:
                                    grp = obj[:3584].reshape(8, 448)[:, :432].reshape(8, 6, 8, 9)  # [j][cb][dim row][9]
                                    vt = decode_rows(np.ascontiguousarray(grp.reshape(8 * 6 * 8, 9)[None]).reshape(1, -1, 9)).reshape(8, 6, 8, 8)  # [j][cb][dim][key]
                                    qkv_dec.setdefault("vT", np.zeros((COLS * M, 768)))[rows, h * 64:(h + 1) * 64] = vt.transpose(1, 3, 0, 2).reshape(48, 64)
                    if kind < 2:
                        got_dec = decode_rows(np.ascontiguousarray(got_rows)).astype(np.float64)
                        exp_dec = decode_rows(exp_u8).astype(np.float64)
                        qkv_dec[name] = got_dec
                        mis, _ = report(f"{op['name']} {name} objects", got_dec, exp_dec)
                        bad = np.mean((got_rows != exp_u8).reshape(-1, 9).any(axis=1)[: M_REAL * 96])
                        print(f"      panels differing {bad:.4%}")
                        rc |= bool(bad > 0.01)
                    else:  # V^T panels share an exponent over 8 keys of one dim: reference = encode the transposed matrix
                        vt_exp = np.zeros((COLS * M, 768))
                        for h in range(12):
                            sl = slice(h * 64, h * 64 + 64)
                            vt_exp[:, sl] = decode_rows(P.encode_hw(np.ascontiguousarray(y[:, 1536 + h * 64: 1536 + h * 64 + 64].T))).T
                        exp_dec = vt_exp
                        mis, _ = report(f"{op['name']} v^T objects", qkv_dec["vT"], exp_dec)
                        rc |= bool(mis > 0.05)
                        if os.environ.get("DBG"):
                            bad = (qkv_dec["vT"][:M_REAL] != exp_dec[:M_REAL])
                            print("      v^T mismatch by key block (row % 48 // 8):", [f"{bad[(np.arange(M_REAL) % 48) // 8 == cb].mean():.3f}" for cb in range(6)])
                            print("      v^T mismatch by dim block (col % 64 // 8):", [f"{bad[:, (np.arange(768) % 64) // 8 == j].mean():.3f}" for j in range(8)])
                attn_in = qkv_dec
            else:
                print(f"SKIP check {op['name']} {op['epi']}")
        elif op["kind"] == "attn":
            q, k, v = attn_in["q"], attn_in["k"], attn_in["vT"]
            o_ref = np.zeros((COLS * M, 768))
            nkeys = int(os.environ.get("ATTN_KEYS", str(M_REAL)))
            for h in range(12):
                sl = slice(h * 64, h * 64 + 64)
                s = q[:, sl] @ k[:nkeys, sl].T / 8.0  # bf16-rounded scores on the device; here float
                s = s - s.max(axis=1, keepdims=True)
                pr = np.exp(s)
                o_ref[:, sl] = pr @ v[:nkeys, sl] / pr.sum(axis=1, keepdims=True)
            dst = op["dst"]
            got_u8 = P.panels_from_apan(get(dst), dst, 12)
            got = decode_rows(np.ascontiguousarray(got_u8)).astype(np.float64)
            err = got[:M_REAL] - o_ref[:M_REAL]
            print(f"CHECK attn O: rms {np.sqrt((err ** 2).mean()):.3e} (ref rms {np.sqrt((o_ref[:M_REAL] ** 2).mean()):.3e}) max|err| {np.abs(err).max():.3e}"
                  f" rms_rel {np.sqrt((err ** 2).sum() / (o_ref[:M_REAL] ** 2).sum()):.3e}")
            rc |= bool(np.sqrt((err ** 2).sum() / (o_ref[:M_REAL] ** 2).sum()) > 0.05)
            if os.environ.get("DBG"):
                g, e = got[:M_REAL], o_ref[:M_REAL]
                rr = lambda a, b: np.sqrt(((a - b) ** 2).sum() / (b ** 2).sum())
                print("      rms_rel by head:", [f"{rr(g[:, h*64:(h+1)*64], e[:, h*64:(h+1)*64]):.2f}" for h in range(12)])
                rows = np.arange(M_REAL)
                print("      rms_rel by query tile (row%144//48):", [f"{rr(g[rows % 144 // 48 == t], e[rows % 144 // 48 == t]):.2f}" for t in range(3)])
                print("      rms_rel by column:", [f"{rr(g[rows // 144 == c], e[rows // 144 == c]):.2f}" for c in range(8)])
                corr = np.corrcoef(g.ravel(), e.ravel())[0, 1]
                print(f"      corr {corr:.3f}; got mean {g.mean():.3f} std {g.std():.3f}; ref mean {e.mean():.3f} std {e.std():.3f}")
                # hypotheses: heads permuted? compare each got head with every ref head
                best = []
                for h in range(12):
                    cs = [np.corrcoef(g[:, h*64:(h+1)*64].ravel(), e[:, k*64:(k+1)*64].ravel())[0, 1] for k in range(12)]
                    best.append((int(np.argmax(cs)), round(float(max(cs)), 2)))
                print("      best matching ref head per got head:", best)
                np.savez(str(HERE / "build" / "t4_dbg.npz"), got=got, q=q, k=k, vT=v, o_ref=o_ref)
            apan[(dst["base"], 0)] = got
        elif op["kind"] == "ew" and op["fn"] == "ssm":
            area = op["area"]
            base = area["base"]

            def slot(k):
                return np.concatenate([ACT_out[(base + c * 8 * S + k * S) // 4: (base + c * 8 * S + (k + 1) * S) // 4] for c in range(COLS)])

            zr = np.zeros((COLS * M, 768), bfloat16)
            hr = np.zeros((COLS * M, 768), np.float32)
            for q in range(12):
                sl = slice(q * 64, q * 64 + 64)
                zq, hq = P.ssm_ref(ssm["y"][:, sl], ssm["x"][:, sl], ssm["xm1"][:, sl], ssm["xm2"][:, sl], ssm["xm3"][:, sl], ssm["h0"][:, sl],
                                   ssm["conv_w"][:, sl], ssm["conv_b"][sl], ssm["a_param"][sl], ssm["wx"][q], ssm["bx"][sl], ssm["wa"][q], ssm["ba"][sl], ssm["first"])
                zr[:, sl], hr[:, sl] = zq, hq
            report("ssm C0 (= x)", P.from_q16(slot(2)), ssm["x"])
            report("ssm C1 (= old C0)", P.from_q16(slot(3)), ssm["xm1"])
            report("ssm C2 (= old C1)", P.from_q16(slot(4)), ssm["xm2"])
            hgot = P.h_from_units(slot(5), slot(6))
            report("ssm h (fp32)", hgot, hr)
            got_p = P.panels_from_q16(slot(7), pan_off=0)
            exp_p = P.encode_hw(f32(zr))
            got_dec = decode_rows(np.ascontiguousarray(got_p)).astype(np.float64)
            exp_dec = decode_rows(exp_p).astype(np.float64)
            mis, ulp = report("ssm z panels", got_dec, exp_dec, ulp_of=np.abs(exp_dec).max(axis=1, keepdims=True) * np.ones_like(exp_dec))
            bad_panels = np.mean((got_p != exp_p).reshape(-1, 9).any(axis=1)[: M_REAL * 96])
            print(f"      panels differing {bad_panels:.4%}")
            rc |= bool(bad_panels > 0.2)
            panels[area["z"]["base"]] = got_dec
        elif op["kind"] == "ew" and op["fn"] == "patch":
            a = decode_rows(P.encode_hw(f32(patch["pb"]))).astype(np.float64)  # the kernel converts the bf16 patches to bfp
            w_dec = decode_rows(P.encode_rows(patch["wf"].reshape(768, 192))).astype(np.float64)
            ref = (a @ w_dec.T + f32(bfv(patch["bfold"]))).astype(np.float32).astype(bfloat16)  # the device adds the bf16 bias
            got = P.from_q16(get(op["dst"]))
            mis, ulp = report("patch embedding", got, ref)
            rc |= bool(mis > 0.05)
            xprime[op["dst"]["base"]] = got
            xprime[info["pos"]["base"]] = patch["pos"]
        elif op["kind"] == "ew" and op["fn"] == "norm":
            srcs = sorted(op["srcs"], key=lambda s: s["base"])
            xs = [xprime[s["base"]] for s in srcs]
            xsum = sum(x.astype(np.float32) for x in xs).astype(bfloat16)
            words = get(op["dst"])
            gotx = P.from_q16(words)
            if op["mode"] != "ln_out":
                mis, _ = report(f"{op['name']} X'", gotx, xsum)
                rc |= bool(mis > 0)
            if op["mode"] == "rms":
                z = P.rmsnorm_ref(xsum, sc1)
            elif op["mode"] in ("ln", "ln_out"):
                z = P.layernorm_ref(xsum, gamma, beta)
            else:
                z = xsum
            exp_p = P.encode_hw(f32(z))
            got_p = P.panels_from_q16(words)
            got_dec = decode_rows(np.ascontiguousarray(got_p)).astype(np.float64)
            exp_dec = decode_rows(exp_p).astype(np.float64)
            mis, ulp = report(f"{op['name']} {op['mode']} panels", got_dec, exp_dec, ulp_of=np.abs(exp_dec).max(axis=1, keepdims=True) * np.ones_like(exp_dec))
            bad_panels = np.mean((got_p != exp_p).reshape(-1, 9).any(axis=1)[: M_REAL * 96])
            print(f"      panels differing {bad_panels:.4%}")
            rc |= bool(bad_panels > (0.01 if op["mode"] == "rms" else 0.2))
            xprime[op["dst"]["base"]] = gotx
            panels[op["dst"]["base"]] = got_dec
            if op["mode"] == "ln_out":
                report(f"{op['name']} ln_out X'", gotx, z)
    return int(rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sched", "run"])
    ap.add_argument("name")
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--timeout-ms", type=int, default=3000)
    o = ap.parse_args()
    if o.cmd == "sched":
        write_sched(o.name)
        return 0
    return run(o.name, o)


if __name__ == "__main__":
    sys.exit(main())
