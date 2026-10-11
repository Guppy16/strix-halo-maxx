"""TAPNext++ step on the NPU as ONE dispatch of a design8 build, with the real weights.

Runs in npu-iron-torch (pyxrt + torch CPU); mounts: /work = .scratch, /repo = the app checkout (read-only).
The host does: patch embedding + pos-emb (torch bf16 conv on the CPU) and query tokens -> X0, one dispatch, the heads.

    python3 step8.py smoke  --build full12a [--frames 4]    NPU vs CPU bf16 on a real clip: tokens, tracks
    python3 step8.py blocks --build full1a                  depth-1 build: every region vs the CPU block
    python3 step8.py davis  --build full12a [--videos 10]   TAP-Vid DAVIS 'first' AJ
    python3 step8.py time   --build full12a [--iters 20]    step latency (dispatch, host parts)
"""

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from ml_dtypes import bfloat16

sys.path.insert(0, "/opt/xilinx/xrt/python")
sys.path.insert(0, "/work/npu-tapnext")
import pyxrt  # noqa: E402

import pack8 as P  # noqa: E402
from plan8 import COLS, M, MAX_DEPTH, OBJ, S, RB, SSM_PARAM_BYTES, UNIT, XB, KSTEP16, full_step  # noqa: E402

HERE = "/work/npu-tapnext"
REPO = "/repo"
tn = types.ModuleType("tn")  # the vendored model targets Python 3.14 (lazy annotations); the container is 3.13
exec(open(f"{REPO}/agent/app/jobs/tapnext/model.py").read().replace("-> TAPNext:", '-> "TAPNext":'), tn.__dict__)
N_IMG = 1024
HOST_FAST = int(os.environ.get("HOST_FAST", "2"))  # round 11: 0 = round-10 host packing, 1 = torch packing, 2 = into the mapped bo
HEAD_THREADS = int(os.environ.get("HEAD_THREADS", "0"))
HOST_HEADS = os.environ.get("HOST_HEADS", "torch")  # round 13: "np" = Step8.heads_np (research H3)  # round 11: torch threads for the heads (0 = leave as is)


def f32(t):
    return t.detach().float().numpy() if isinstance(t, torch.Tensor) else np.asarray(t, np.float32)


def load_model(depth):
    model = tn.TAPNext()
    tn.load_state(model, f"{REPO}/data/models/tapnextpp_256.safetensors")
    model.to_half(torch.bfloat16).eval()
    model.blocks = model.blocks[:depth]
    return model


def col_rows(words, nq=12, unit=UNIT, xoff=0):
    """One column of a Q16 region ([q][rb][unit] bytes) -> its 144 rows [144, 64 nq] bf16."""
    u = words.view(np.uint8).reshape(nq, RB, unit)[..., xoff:xoff + XB].view(bfloat16).reshape(nq, RB, 8, 8, 8)
    return np.ascontiguousarray(u.transpose(1, 3, 0, 2, 4)).reshape(M, nq * 64)


def q16_col(x144):
    """[144, 768] bf16 rows of one column -> the X' parts of that column's Q16 slice (panel parts zero)."""
    out = np.zeros((12, RB, UNIT), np.uint8)
    v = np.asarray(x144, bfloat16).reshape(RB, 8, 12, 8, 8).transpose(2, 0, 3, 1, 4)  # [q][rb][cb][row][col]
    out[..., :XB] = np.ascontiguousarray(v).reshape(12, RB, XB // 2).view(np.uint8)
    return out.reshape(-1).view(np.uint32)


class Step8:
    """The NPU step with the real weights. `build` names any depth's build of one design variant (e.g. full12h);
    every depth of that variant shares the xclbin, the memory layout and the packed W (plan8.full_step allocates
    for MAX_DEPTH blocks), so `set_depth(d)` only swaps the runtime sequence: build/full<d><suffix>.bin."""

    def __init__(self, build, depth, n_slots=48):
        self.n_slots = n_slots
        self.build = build
        import plan8  # the plan options a build was made with (design8 records them in build/<name>.json)

        bj = json.load(open(f"{HERE}/build/{build}.json"))
        plan8.A_HALF, plan8.P23, plan8.P60G = bool(bj.get("a_half", False)), bool(bj.get("p23", False)), bool(bj.get("p60g", False))
        # round 17 (X1 full rotation): XROT = 4 frame phases, each its own runtime sequence (build/full<d><suffix>_p<k>.bin, k >= 1)
        plan8.XROT, plan8.XPH = int(bj.get("xrot", 0)), 0
        plan8.IL = bool(bj.get("il", False))  # round 20: mlp0 rr + split drain, mlp3_a half 1 first
        plan8.DAPRE = bool(bj.get("dapre", False))  # round 20: down_a half 1 first
        plan8.A3 = bool(bj.get("a3", False))  # round 22 3A: down_b / mlp3_b fills split for region 3
        plan8.YXR = bool(bj.get("yxr", False))  # round 23 YXR: yx's steps interleaved y / x
        plan8.YXH = bool(bj.get("yxh", False))  # round 24 YXH: yx's steps head-major
        plan8.ZDROP = bool(bj.get("zdrop", False))  # round 24 ZDROP: SRES SSMs drain h only
        plan8.MSPLIT = tuple(bj.get("msplit", []))  # round 23 MSPLIT-P: the last tile in two row passes
        plan8.NRES = tuple(bj.get("nres", []))
        plan8.SRES = bool(bj.get("sres", False))  # round 20: norms whose panels go on chip to the next GEMM's A slot
        plan8.AL1 = bool(bj.get("al1", False))  # round 17: params fills padded to 1 KB (the per-head SSM stride: op["pstride"])
        self.nph, self.phase = max(plan8.XROT, 1), 0
        self.p_all, self.info_all = full_step(MAX_DEPTH)  # every block: the weights are packed once for all depths
        meta = json.load(open(f"{HERE}/build/{build}.sched.meta"))
        P.B_PAIR4 = P.uses_asm_gemm(f"{HERE}/build/{build}.json")
        P.B_ROT = P.rot_order(f"{HERE}/build/{build}.json")  # round 18 ROT
        assert meta["act_words"] == max(self.p_all.act_bytes, 64) // 4 and meta["w_words"] == max(self.p_all.w_bytes, 64) // 4, ("build/plan mismatch", meta)
        self.model = load_model(MAX_DEPTH)
        t0 = time.perf_counter()
        self.W = np.zeros(meta["w_words"], np.uint32)
        self.ACT = np.zeros(meta["act_words"], np.uint32)
        self._pack()
        print(f"packed W {self.W.nbytes / 1e6:.0f} MB in {time.perf_counter() - t0:.1f} s", flush=True)
        dev = self.dev = pyxrt.device(0)
        b = f"{HERE}/build/{build}"
        self.xb = pyxrt.xclbin(b + ".xclbin")
        dev.register_xclbin(self.xb)
        self.ctx = pyxrt.hw_context(dev, self.xb.get_uuid())
        self.kern = pyxrt.kernel(self.ctx, self.xb.get_kernels()[0].get_name())
        self.bo_w = pyxrt.bo(dev, self.W.nbytes, pyxrt.bo.host_only, self.kern.group_id(3))
        self.bo_a = pyxrt.bo(dev, self.ACT.nbytes, pyxrt.bo.host_only, self.kern.group_id(4))
        self.bo_w.write(self.W.tobytes(), 0)
        self.bo_a.write(self.ACT.tobytes(), 0)
        for bo in (self.bo_w, self.bo_a):
            bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self.instr = {}  # (depth, phase) -> (bo, n words, plan, info)
        self.first = True
        self.last_us = 0.0
        self.set_depth(depth)

    def seq_name(self, depth, phase=0):
        """The build whose runtime sequence runs `depth` blocks: full<depth><suffix> next to `build` (round 17: + _p<phase>)."""
        m = __import__("re").match(r"full(\d+)(.*)", self.build)
        return f"full{depth}{m.group(2)}" + (f"_p{phase}" if phase else "")

    def set_depth(self, depth):
        """Run `depth` blocks from the next step on (no rebuild, no new hw context: another instruction stream).
        Blocks that were not running hold stale recurrent state: reset() before tracking on with a larger depth."""
        for k in range(self.nph):
            if (depth, k) in self.instr:
                continue
            name = self.seq_name(depth, k)
            p, info = full_step(depth, phase=k)
            sched = f"{HERE}/build/{name}.sched"
            assert open(sched).read() == p.sched_json(), f"stale build {name}: plan8 changed since"
            instr = np.fromfile(f"{HERE}/build/{name}.bin", dtype=np.uint32)
            bo = pyxrt.bo(self.dev, instr.nbytes, pyxrt.bo.cacheable, self.kern.group_id(1))
            bo.write(instr.tobytes(), 0)
            bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self.instr[depth, k] = (bo, len(instr), p, info)
        self.depth = depth
        self.select(self.phase)

    def select(self, phase):
        """round 17: the runtime sequence (and its plan) of frame phase `phase` for the next dispatch"""
        self.bo_i, self.n_instr, self.p, self.info = self.instr[self.depth, phase]
        self.phase = phase

    # ---- packing -------------------------------------------------------------------------------------------------
    def _pack(self):
        W = self.W
        sd = {k: f32(v) for k, v in self.model.state_dict().items()}

        def put_w(off, words):
            W[off // 4: off // 4 + words.size] = words

        for op in self.p_all.ops:
            if op["kind"] == "gemm":
                b, name = op["name"].split("_", 1)
                pre = f"blocks.{int(b[1:])}."
                rb_, mb_, vb_ = pre + "ssm_block.recurrent_block.", pre + "ssm_block.mlp_block.", pre + "vit_block."
                if name == "yx":
                    wt, bias = sd[rb_ + "linear_yx.weight"], sd[rb_ + "linear_yx.bias"]
                elif name == "out":
                    wt, bias = sd[rb_ + "linear_out.weight"], sd[rb_ + "linear_out.bias"]
                elif name in ("up_a", "up_b"):
                    w, bb = sd[mb_ + "ffw_up.weight"], sd[mb_ + "ffw_up.bias"]
                    wt, bias = P.interleave_gated(w[:3072], w[3072:], bb[:3072], bb[3072:])
                    h = slice(0, 3072) if name == "up_a" else slice(3072, 6144)
                    wt, bias = wt[h], bias[h]
                elif name in ("down_a", "down_b"):
                    w = sd[mb_ + "ffw_down.weight"]
                    wt = w[:, :1536] if name == "down_a" else w[:, 1536:]
                    bias = sd[mb_ + "ffw_down.bias"]
                elif name == "qkv":
                    wt, bias = sd[vb_ + "self_attention.in_proj_weight"], sd[vb_ + "self_attention.in_proj_bias"]
                elif name == "out_proj":
                    wt, bias = sd[vb_ + "self_attention.out_proj.weight"], sd[vb_ + "self_attention.out_proj.bias"]
                elif name == "mlp0":
                    wt, bias = sd[vb_ + "mlp.0.weight"], sd[vb_ + "mlp.0.bias"]
                elif name in ("mlp3_a", "mlp3_b"):
                    w = sd[vb_ + "mlp.3.weight"]
                    wt = w[:, :1536] if name == "mlp3_a" else w[:, 1536:]
                    bias = sd[vb_ + "mlp.3.bias"]
                else:
                    raise KeyError(op["name"])
                put_w(op["w"], P.pack_gemm(op, np.ascontiguousarray(wt), bias))
                # timing experiments: EPI_OVERRIDE="mlp0:0,yx:0" rewrites the epilogue flag of every tile of those ops
                for item in filter(None, os.environ.get("EPI_OVERRIDE", "").split(",")):
                    nm, flag = item.split(":")
                    if name == nm:
                        for pos in range(3 * op["nt"]):
                            W[(op["w"] + (pos + 1) * op["lw"] - OBJ + 1024) // 4] = int(flag)
            elif op.get("fn") == "norm":
                if op["name"] == "final":
                    put_w(op["params_w"], P.norm_params("ln_out", sd["encoder_norm.weight"], sd["encoder_norm.bias"], n_in=op["n_in"], eps=self.model.encoder_norm.eps))
                    continue
                b, name = op["name"].split("_", 1)
                pre = f"blocks.{int(b[1:])}."
                if name in ("norm0", "norm1"):
                    key = pre + "ssm_block." + ("temporal_pre_norm" if name == "norm0" else "channel_pre_norm") + ".scale"
                    put_w(op["params_w"], P.norm_params("rms", sd[key], n_in=op["n_in"], nres=op.get("nres", False)))
                else:
                    ln = pre + "vit_block." + ("ln_1" if name == "norm2" else "ln_2")
                    put_w(op["params_w"], P.norm_params("ln", sd[ln + ".weight"], sd[ln + ".bias"], n_in=op["n_in"], nres=op.get("nres", False)))
            elif op.get("fn") == "ssm":  # (round 17: XROT also packs the four phases' copies, header word 2 = 1 + phase)
                pre = f"blocks.{int(op['name'].split('_')[0][1:])}.ssm_block.recurrent_block."
                cw, cb = sd[pre + "conv_1d.w"], sd[pre + "conv_1d.b"]
                ap = sd[pre + "rg_lru.a_param"]
                wx, bx = sd[pre + "rg_lru.input_gate.w"], sd[pre + "rg_lru.input_gate.b"].reshape(-1)
                wa, ba = sd[pre + "rg_lru.a_gate.w"], sd[pre + "rg_lru.a_gate.b"].reshape(-1)
                for q in range(12):
                    sl = slice(q * 64, q * 64 + 64)
                    prm = P.ssm_params(q, cw[:, sl], cb[sl], ap[sl], wx[q], bx[sl], wa[q], ba[sl], 1)
                    if op.get("sres"):  # round 20 SRES: the z panels to the EW core's panel ring
                        prm[3] |= 8
                    ps = op.get("pstride", SSM_PARAM_BYTES)
                    put_w(op["params_w"] + q * ps, prm)
                    for r in range(4 if op.get("params_rot") is not None else 0):
                        prm = prm.copy()
                        prm[2] = 1 + r
                        put_w(op["params_rot"] + (r * 12 + q) * ps, prm)

    # ---- per-frame host work --------------------------------------------------------------------------------------
    def _sync_to(self, bo, byte_off, nbytes):
        bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, nbytes, byte_off)

    def _write_act(self, byte_off, words):
        self.bo_a.write(words.tobytes(), byte_off)
        self._sync_to(self.bo_a, byte_off, words.nbytes)

    def set_first(self, first):
        """The SSM ops' `first` header word (RG-LRU without state, zero conv cache) for every block and head."""
        one = np.array([1 if first else 0], np.uint32)
        for op in self.p_all.ops:
            if op["kind"] == "ew" and op["fn"] == "ssm":
                for q in range(12):
                    ps = op.get("pstride", SSM_PARAM_BYTES)
                    for off in [op["params_w"] + q * ps] + ([op["params_rot"] + (r * 12 + q) * ps for r in range(4)] if op.get("params_rot") is not None else []):
                        self.bo_w.write(one.tobytes(), off)
                        self._sync_to(self.bo_w, off, 64)
        self.first = first

    def reset(self):
        """A new video: zero every block's SSM area (conv cache and h) and mark the next frame first."""
        for a in self.info_all["areas"]:
            self._write_act(a.get("area", a["base"]), np.zeros(a["span"] // 4, np.uint32))
        self.set_first(True)
        self.select(0)  # round 17: a video starts at phase 0 (the state at the area's top)

    def x0_rows(self, frame, queries):
        """X0 = [patch embedding + pos-emb (1024 image rows); query tokens; zero padding] as [COLS*M, 768] bf16,
        computed on the CPU exactly as torch does in bf16."""
        m = self.model
        with torch.no_grad():
            x = m.lin_proj(frame.permute(2, 0, 1)[None].to(torch.bfloat16))[0].flatten(1).transpose(0, 1) + m.image_pos_emb[0]
            tok = m.embed_queries(queries).to(x.dtype)
        if not HOST_FAST:
            rows = torch.zeros(COLS * M, 768, dtype=torch.bfloat16)
        else:  # round 11: persistent; rows past the queries stay zero (a step never has more queries than slots)
            if getattr(self, "_rows", None) is None:
                self._rows = torch.zeros(COLS * M, 768, dtype=torch.bfloat16)
            rows = self._rows
            rows[N_IMG + tok.shape[0]:] = 0
        rows[:N_IMG] = x
        rows[N_IMG:N_IMG + tok.shape[0]] = tok
        return rows

    def write_x0(self, rows):
        """The X' parts of the X0 region (Q16 layout) <- rows. HOST_FAST (round 11, r11/q16bench.py): torch's strided
        copy into a persistent buffer whose panel parts stay zero (0.14 vs 0.75 ms, byte-identical); HOST_FAST=2 packs
        straight into the mapped ACT bo (no bo.write copy)."""
        base = self.info["x0"]["base"]
        if not HOST_FAST:
            self._write_act(base, P.to_q16(rows.view(torch.int16).numpy().view(bfloat16)))
            return
        n16 = COLS * 12 * RB * UNIT // 2
        if getattr(self, "_x0buf", None) is None:
            if HOST_FAST == 2:
                self._amap = torch.frombuffer(self.bo_a.map(), dtype=torch.int16)
                self._x0buf = self._amap[base // 2:base // 2 + n16].view(COLS, 12, RB, UNIT // 2)
                self._x0buf[..., XB // 2:] = 0
            else:
                self._x0buf = torch.zeros((COLS, 12, RB, UNIT // 2), dtype=torch.int16)
        v = rows.view(torch.int16).reshape(COLS, RB, 8, 12, 8, 8).permute(0, 3, 1, 4, 2, 5)
        self._x0buf[..., :XB // 2].view(COLS, 12, RB, 8, 8, 8).copy_(v)
        if HOST_FAST == 2:
            self._sync_to(self.bo_a, base, n16 * 2)
        else:
            self._write_act(base, self._x0buf.numpy().reshape(-1).view(np.uint32))

    def put_x0(self, frame, queries):
        self.write_x0(self.x0_rows(frame, queries))

    def read_col(self, region, col=7, nq=12, xoff=0):
        off, n = region["base"] + col * region["col"], nq * KSTEP16
        self.bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, n, off)
        return col_rows(np.frombuffer(self.bo_a.read(n, off), np.uint32), nq=nq, xoff=xoff)

    def read_all(self):
        self.bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        return np.frombuffer(self.bo_a.read(self.ACT.nbytes, 0), np.uint32)

    def dispatch(self, timeout_ms=3000):
        t0 = time.perf_counter()
        r = self.kern(3, self.bo_i, self.n_instr, self.bo_w, self.bo_a)
        st = r.wait(timeout_ms * 1000)
        self.last_us = (time.perf_counter() - t0) * 1e6
        self.last_phase = self.phase
        if self.nph > 1:  # round 17: the next frame's phase (self.p / self.info stay this frame's until then)
            self.select((self.phase + 1) % self.nph)
            self.p, self.info = self.instr[self.depth, self.last_phase][2:]
        if str(st) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
            raise RuntimeError(f"dispatch failed: {st} after {self.last_us:.0f} us")

    def step(self, frame, queries):
        """frame [256, 256, 3] in [-1, 1]; queries [1, Q, 3] (t relative to this frame). Returns tracks, logits, vis
        and the encoder-normed query tokens."""
        self.put_x0(frame, queries)
        self.dispatch()
        if self.first:
            self.set_first(False)
        q = queries.shape[1]
        x = self.read_col(self.info["final"])[16:16 + q]
        return (*self.heads(x), x)

    def heads_np(self, x):
        """Round 13 (npu-research H3): model.heads as plain numpy fp32 (the torch version is ~40 small ops of overhead
        around ~20 MFLOP). Each head: Linear 768 -> 256, LayerNorm, erf GELU, Linear, LayerNorm, GELU, Linear; then the
        masked softmax over +-20 of the argmax at temperature 0.5, tracks = expectation + 0.5 (y, x); vis = the logit."""
        if not hasattr(self, "_hw"):
            def lin(mod):
                return np.ascontiguousarray(f32(mod.weight).T), f32(mod.bias)
            self._hw = [[(lin(h[0]), (f32(h[1].weight), f32(h[1].bias), h[1].eps), lin(h[3]), (f32(h[4].weight), f32(h[4].bias), h[4].eps),
                          lin(h[6]))] for h in (self.model.coordinate_head, self.model.visible_head)]
            self._erf = lambda a: torch.erf(torch.from_numpy(np.ascontiguousarray(a))).numpy()  # numpy has no erf (no scipy here)

        def mlp(x, p):
            (w0, b0), (g0, be0, e0), (w1, b1), (g1, be1, e1), (w2, b2) = p
            for w, b, g, be, e in ((w0, b0, g0, be0, e0), (w1, b1, g1, be1, e1)):
                h = x @ w + b
                mu = h.mean(-1, keepdims=True)
                var = ((h - mu) ** 2).mean(-1, keepdims=True)
                h = (h - mu) / np.sqrt(var + e) * g + be
                x = 0.5 * h * (1.0 + self._erf(h * np.float32(0.7071067811865476)))
            return x @ w2 + b2

        x = x.astype(np.float32)
        logits = mlp(x, self._hw[0][0])
        n = logits.shape[-1] // 2
        idx = np.arange(n, dtype=np.float32)
        out = []
        for h in (logits[:, :n], logits[:, n:]):
            am = h.argmax(-1, keepdims=True)
            mask = np.abs(am - idx) <= 20
            z = h * np.float32(0.5)
            e = np.exp(z - z.max(-1, keepdims=True))
            pr = e / e.sum(-1, keepdims=True) * mask
            pr = pr / pr.sum(-1, keepdims=True)
            out.append((pr * idx).sum(-1))
        tracks = np.stack(out, -1) + 0.5
        return tracks, logits, mlp(x, self._hw[1][0])[:, 0]

    def heads(self, x):
        """The tracking heads on the CPU. HEAD_THREADS (round 11, r11/hostprof.py): the heads are small; with 8 threads
        they took 0.68-1.23 ms (median 0.87), on one thread a steady 0.83. HOST_HEADS=np (round 13): heads_np."""
        if HOST_HEADS == "np":
            return self.heads_np(x)
        nt = torch.get_num_threads()
        if HEAD_THREADS:
            torch.set_num_threads(HEAD_THREADS)
        with torch.no_grad():
            out = self.model.heads(torch.from_numpy(x.astype(np.float32)))
        if HEAD_THREADS:
            torch.set_num_threads(nt)
        return out


def preprocess(frames_u8, res=256):
    import torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(frames_u8)).permute(0, 3, 1, 2).float()
    t = F.interpolate(t, size=(res, res), mode="bilinear", align_corners=False)
    return (t / 127.5 - 1.0).permute(0, 2, 3, 1).contiguous()


def cpu_step_tokens(model, frame, queries, caches):
    """The CPU bf16 step, also returning the encoder-normed query tokens."""
    out = {}
    h = model.encoder_norm.register_forward_hook(lambda m, i, o: out.__setitem__("x", o))
    try:
        tracks, logits, vis, caches = model.step(frame, queries, caches)
    finally:
        h.remove()
    return tracks, logits, vis, caches, f32(out["x"])


def demo_queries(n=48):
    q = [[0.0, 100.0, 100.0], [0.0, 200.0, 60.0], [0.0, 50.0, 220.0], [0.0, 128.0, 128.0]]
    return torch.tensor([q + [[10000.0, 0.0, 0.0]] * (n - len(q))])


def smoke(o):
    frames = np.load("/work/tapnet/data/clip30.npy")[: o.frames]
    video = preprocess(frames)
    s = Step8(o.build, o.depth)
    ref = load_model(o.depth)
    q = demo_queries()
    s.reset()
    caches = None
    with torch.no_grad():
        for t in range(len(video)):
            rel = torch.cat([q[..., :1] - t, q[..., 1:]], -1)
            tr_n, _, vis_n, x_n = s.step(video[t], rel)
            tr_c, _, vis_c, caches, x_c = cpu_step_tokens(ref, video[t], rel, caches)
            xq = x_c[N_IMG:N_IMG + 48]
            d = np.abs(x_n.astype(np.float32) - xq)
            corr = np.corrcoef(x_n.astype(np.float32).ravel(), xq.ravel())[0, 1]
            print(f"frame {t}: dispatch {s.last_us / 1e3:.1f} ms; tokens corr {corr:.5f} rms_rel {np.sqrt((d ** 2).sum() / (xq ** 2).sum()):.3e}"
                  f" max {d.max():.3f}; tracks max|d| {(tr_n[:4] - tr_c[:4]).abs().max():.3f} px", flush=True)
            print(f"   NPU {tr_n[:4].numpy().round(2).tolist()} vis {vis_n[:4].numpy().round(2).tolist()}")
            print(f"   CPU {tr_c[:4].numpy().round(2).tolist()} vis {vis_c[:4].numpy().round(2).tolist()}")


def regress(o):
    """Bit-level regression between builds (round 6): run o.frames frames of clip30 and save (--save NAME) or compare
    (--ref NAME) the final LN output of all 1152 rows and the tracks. A refactor that keeps the numerics must give
    0 differing words; a numeric change is summarised as the fraction of differing tokens and the largest differences."""
    frames = np.load("/work/tapnet/data/clip30.npy")[: o.frames]
    video = preprocess(frames)
    s = Step8(o.build, o.depth)
    q = demo_queries()
    s.reset()
    toks, trs = [], []
    for t in range(len(video)):
        rel = torch.cat([q[..., :1] - t, q[..., 1:]], -1)
        tr, _, _, _ = s.step(video[t], rel)
        toks.append(np.concatenate([s.read_col(s.info["final"], col=c) for c in range(COLS)]))
        trs.append(tr[:4].numpy())
    toks, trs = np.stack(toks).view(np.uint16), np.stack(trs)  # raw bf16 bits (npz does not keep ml_dtypes)
    path = Path("/work/npu-tapnext/build") / f"regress_{o.save or o.ref}_d{o.depth}.npz"
    if o.save:
        np.savez(path, toks=toks, trs=trs)
        print(f"REGRESS saved {path.name}: {toks.shape} tokens, tracks {trs.shape}")
        return
    ref = np.load(path)
    a, b = toks.view(bfloat16).astype(np.float32), ref["toks"].view(bfloat16).astype(np.float32)
    for t in range(len(a)):
        d = np.abs(a[t] - b[t])
        print(f"REGRESS frame {t}: differing tokens {np.mean(a[t] != b[t]):.4%} max|d| {d.max():.4f} rms_rel "
              f"{np.sqrt((d ** 2).sum() / (b[t] ** 2).sum()):.3e}; tracks max|d| {np.abs(trs[t] - ref['trs'][t]).max():.4f} px")
    print("REGRESS", "BIT-IDENTICAL" if np.array_equal(toks, ref["toks"]) else "DIFFERENT", f"vs {o.ref} ({o.build}, depth {o.depth})")


def blocks(o):
    """depth-1 build: compare every region of block 0 against the CPU block (first frame, then a second frame)."""
    assert o.depth == 1
    frames = np.load("/work/tapnet/data/clip30.npy")[:2]
    video = preprocess(frames)
    s = Step8(o.build, 1)
    m = load_model(1)
    q = demo_queries()
    s.reset()
    caches = None
    r = s.info["regions"]
    for t in range(2):
        rel = torch.cat([q[..., :1] - t, q[..., 1:]], -1)
        s.step(video[t], rel)
        act = s.read_all()
        with torch.no_grad():
            x = m.lin_proj(video[t].permute(2, 0, 1)[None].to(torch.bfloat16))[0].flatten(1).transpose(0, 1) + m.image_pos_emb[0]
            x = torch.cat([x, m.embed_queries(rel).to(x.dtype)])
            blk = m.blocks[0]
            sb = blk.ssm_block
            t0 = sb.temporal_pre_norm(x)
            rec = sb.recurrent_block
            y, xx = rec.linear_yx(t0).chunk(2, dim=-1)
            yg = tn.gelu(y)
            xc, conv = rec.conv_1d(xx, None if caches is None else caches[0].conv)
            xl, lru = rec.rg_lru(xc, None if caches is None else caches[0].lru)
            hh = rec.linear_out(xl * yg)
            res = hh + x
            l1 = sb.channel_pre_norm(res)
            x1 = sb.mlp_block(l1) + res
            vb = blk.vit_block
            a = vb.self_attention(vb.ln_1(x1))
            x2 = x1 + a
            x3 = x2 + vb.mlp(vb.ln_2(x2))
            fin = m.encoder_norm(x3)
            caches = [tn.BlockCache(lru=lru, conv=conv)]
        R = 1072

        def words(region, nq=12):  # the region's per-column slices, concatenated (regions may be strided, e.g. SSM areas)
            return np.concatenate([act[(region["base"] + c * region["col"]) // 4:][: nq * KSTEP16 // 4] for c in range(COLS)])

        def get(region, nq=12):
            return P.from_q16(words(region, nq), ncols=nq * 64)[:R]

        def cmp(label, got, ref):
            g, e = np.asarray(got, np.float32)[:R], f32(ref)[:R]
            err = g - e
            corr = np.corrcoef(g.ravel(), e.ravel())[0, 1]
            print(f"  {label:28s} corr {corr:.5f} rms_rel {np.sqrt((err ** 2).sum() / max((e ** 2).sum(), 1e-30)):.3e} max|err| {np.abs(err).max():.3e}")

        def pan(region, nq=12):
            return P.decode_rows(np.ascontiguousarray(P.panels_from_q16(words(region, nq), ncols=nq * 64, pan_off=region["pan_off"])))

        print(f"frame {t}:")
        cmp("x0 (R0 X')", get(r["R0"]), x)
        cmp("t = rmsnorm (R0 panels)", pan(r["R0"]), t0)
        area = s.info["areas"][0]
        yx = get(area["yx"], nq=24)
        cmp("y gelu (YX y)", yx[:, :768], yg)
        cmp("x (YX x)", yx[:, 768:], xx)
        cmp("z panels", pan(area["z"]), xl * yg)
        cmp("h = linear_out (R1)", get(r["R1"]), hh)
        cmp("residual (R2 X')", get(r["R2"]), res)
        cmp("channel norm (R2 panels)", pan(r["R2"]), l1)
        cmp("x1 (R3 X')", get(r["R3"]), x1)
        cmp("ln_1 (R3 panels)", pan(r["R3"]), vb.ln_1(x1))
        cmp("attn out_proj (R4)", get(r["R4"]), a)
        cmp("x2 (R5 X')", get(r["R5"]), x2)
        cmp("ln_2 (R5 panels)", pan(r["R5"]), vb.ln_2(x2))
        cmp("final LN", get(s.info["final"]), fin)


def davis(o):
    # davis.py imports tnpp (the repo model); only its loader, query sampling and metrics are needed (as davis_cpu.py)
    src = "\n".join(line for line in open("/work/tapnet/davis.py").read().split("\n") if "import tnpp" not in line)
    dv = types.ModuleType("dv")
    dv.__dict__["__file__"] = "/work/tapnet/davis.py"
    exec(src.split("def track_video")[0], dv.__dict__)
    dv.ROOT = __import__("pathlib").Path("/work/tapnet")
    data = dv.load_davis(dv.ROOT / "data" / "tapvid_davis.zip")
    names = sorted(data.keys())[: o.videos]
    s = Step8(o.build, o.depth)
    per_video, ms, kus = [], [], []
    for name in names:
        d = data[name]
        frames = d["video"]
        T = frames.shape[0]
        pts = d["points"] * np.array([256.0, 256.0])
        qp, gt_tracks, gt_occ = dv.sample_queries_first(d["occluded"], pts)
        Q = qp.shape[1]
        qq = qp
        if o.support:  # co-tracked grid points per query (davis_cpu.py's add_support), discarded on output
            qq = dv.add_support(torch.from_numpy(qp.astype(np.float32)), o.support, 8.0).numpy()
        n = qq.shape[1]
        assert n <= o.slots <= 128, (n, o.slots)
        q = torch.from_numpy(qq.astype(np.float32))
        q = torch.cat([q, torch.full((1, o.slots - n, 3), 10000.0)], 1) if n < o.slots else q
        video = preprocess(frames)
        s.reset()
        tr, vl, kk = [], [], []
        t1 = time.perf_counter()
        for t in range(T):
            rel = torch.cat([q[..., :1] - t, q[..., 1:]], -1)
            tracks, _, vis, _ = s.step(video[t], rel)
            kk.append(s.last_us)
            tr.append(tracks[:Q].clone())
            vl.append(vis[:Q].clone())
        ms.append((time.perf_counter() - t1) * 1e3 / T)
        kus.append(float(np.median(kk)))
        tracks = torch.stack(tr).numpy()
        vis = torch.stack(vl).numpy()
        pred_tracks = tracks[..., ::-1].transpose(1, 0, 2)[None]
        pred_occ = ~(vis > 0).T[None]
        m = dv.compute_tapvid_metrics(qp, gt_occ, gt_tracks, pred_occ, pred_tracks)
        per_video.append({k: float(np.mean(v)) for k, v in m.items()})
        print(f"{name:24s} T={T:3d} Q={Q:3d} AJ={per_video[-1]['average_jaccard'] * 100:5.1f} OA={per_video[-1]['occlusion_accuracy'] * 100:5.1f} "
              f"{ms[-1]:.0f} ms/f (dispatch {kus[-1] / 1e3:.1f} ms)", flush=True)
    agg = {k: round(float(np.mean([v[k] for v in per_video])) * 100, 2) for k in per_video[0]}
    row = dict(mode="npu_step8", build=o.build, depth=o.depth, support=o.support, slots=o.slots, videos=len(names), ms_per_frame=round(float(np.median(ms)), 1),
               dispatch_ms=round(float(np.median(kus)) / 1e3, 2), **agg)
    with open(f"{HERE}/davis_npu.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print("RESULT", json.dumps(row))


def timing(o):
    frames = np.load("/work/tapnet/data/clip30.npy")[:2]
    video = preprocess(frames)
    s = Step8(o.build, o.depth)
    q = demo_queries()
    s.reset()
    tk, th, tp = [], [], {}
    for it in range(o.warmup + o.iters):
        t0 = time.perf_counter()
        rel = torch.cat([q[..., :1] - it, q[..., 1:]], -1)
        s.step(video[it % 2], rel)
        if it >= o.warmup:
            th.append((time.perf_counter() - t0) * 1e3)
            tk.append(s.last_us / 1e3)
            tp.setdefault(s.last_phase, []).append(s.last_us / 1e3)
    busy = open("/sys/class/drm/card1/device/gpu_busy_percent").read().strip() if os.path.exists("/sys/class/drm/card1/device/gpu_busy_percent") else "?"
    print(f"TIME {o.build}: dispatch median {np.median(tk):.2f} min {min(tk):.2f} ms; whole step (host incl.) median {np.median(th):.2f} ms; "
          f"iters {o.iters} gpu_busy% {busy}")
    q = np.percentile(tk, [10, 25, 75, 90])  # round 17: the dispatch times are bimodal under the foreign CPU load: show the spread
    print(f"SPREAD {o.build}: p10 {q[0]:.2f} p25 {q[1]:.2f} median {np.median(tk):.2f} p75 {q[2]:.2f} p90 {q[3]:.2f} mean {np.mean(tk):.2f} ms")
    if s.nph > 1:  # round 17: per frame phase (the last one is the wrap frame) and the mean over a cycle of phases
        med = {k: float(np.median(v)) for k, v in sorted(tp.items())}
        print("PHASES " + " ".join(f"p{k} {m:.2f}" for k, m in med.items()) + f"; mean over a cycle {np.mean(list(med.values())):.2f} ms")


def headscheck(o):
    """Round 13 (H3): torch heads vs heads_np on real step outputs: max differences and the time of each."""
    frames = np.load("/work/tapnet/data/clip30.npy")[:2]
    video = preprocess(frames)
    s = Step8(o.build, o.depth)
    q = demo_queries()
    s.reset()
    tt, tn, dt, dl, dv = [], [], 0.0, 0.0, 0.0
    for it in range(o.warmup + o.iters):
        rel = torch.cat([q[..., :1] - it, q[..., 1:]], -1)
        s.put_x0(video[it % 2], rel)
        s.dispatch()
        if s.first:
            s.set_first(False)
        x = s.read_col(s.info["final"])[16:16 + q.shape[1]]
        t0 = time.perf_counter()
        with torch.no_grad():
            a = s.model.heads(torch.from_numpy(x.astype(np.float32)))
        t1 = time.perf_counter()
        b = s.heads_np(x)
        t2 = time.perf_counter()
        if it >= o.warmup:
            tt.append((t1 - t0) * 1e3)
            tn.append((t2 - t1) * 1e3)
        dt = max(dt, float(np.abs(a[0].numpy() - b[0]).max()))
        dl = max(dl, float(np.abs(a[1].numpy() - b[1]).max()))
        dv = max(dv, float(np.abs(a[2].numpy() - b[2]).max()))
    print(f"HEADS {o.build}: torch median {np.median(tt):.3f} p90 {np.percentile(tt, 90):.3f} ms; numpy median {np.median(tn):.3f} "
          f"p90 {np.percentile(tn, 90):.3f} ms; max |d| tracks {dt:.2e} px, logits {dl:.2e}, vis {dv:.2e}; threads {torch.get_num_threads()}")


def sweep(o):
    """One process, one xclbin (--build), several depths via set_depth: dispatch and whole-step times per depth."""
    frames = np.load("/work/tapnet/data/clip30.npy")[:2]
    video = preprocess(frames)
    depths = [int(d) for d in o.depths.split(",")]
    s = Step8(o.build, depths[0])
    q = demo_queries()
    busy = lambda: open("/sys/class/drm/card1/device/gpu_busy_percent").read().strip() if os.path.exists("/sys/class/drm/card1/device/gpu_busy_percent") else "?"
    for d in depths:
        s.set_depth(d)
        s.reset()
        tk, th, tp = [], [], {}
        for it in range(o.warmup + o.iters):
            t0 = time.perf_counter()
            rel = torch.cat([q[..., :1] - it, q[..., 1:]], -1)
            s.step(video[it % 2], rel)
            if it >= o.warmup:
                th.append((time.perf_counter() - t0) * 1e3)
                tk.append(s.last_us / 1e3)
                tp.setdefault(s.last_phase, []).append(s.last_us / 1e3)
        print(f"SWEEP {o.build} depth {d:2d}: dispatch median {np.median(tk):.2f} min {min(tk):.2f} ms; whole step median {np.median(th):.2f} ms; "
              f"iters {o.iters} gpu_busy% {busy()}", flush=True)
        if s.nph > 1:  # round 17: per frame phase and the mean over a cycle (the last phase is the wrap frame)
            med = {k: float(np.median(v)) for k, v in sorted(tp.items())}
            print(f"PHASES depth {d:2d}: " + " ".join(f"p{k} {m:.2f}" for k, m in med.items()) + f"; mean over a cycle {np.mean(list(med.values())):.2f} ms", flush=True)


def tap_ends(t):
    """First and last byte offset a plan8 TAP touches (sizes/strides in bytes, inner stride 1)."""
    last = t["off"] + sum((n - 1) * st for n, st in zip(t["sizes"][:3], t["strides"][:3])) + t["sizes"][3] - 4
    return t["off"], last


def op_marks(op):
    """(first, last) byte offsets of column 0's output of an op: the first and last word its drain writes."""
    c = op["c"]
    if op["kind"] == "attn":
        return tap_ends(c[0][0])[0], tap_ends(c[-1][0])[1]
    c0 = c[0]
    if isinstance(c0, list):  # qkv: 12 drains per column
        return tap_ends(c0[0])[0], tap_ends(c0[-1])[1]
    return tap_ends(c0)


def pass1_mark(op):
    """round 24: a split op's (MSPLIT) last tile, GEMM row 2, row block 5's last word: pass 1 writes it with its real value and pass 2
    rewrites the same value, so it changes once per block, at the end of pass 1's drain."""
    t = op["c"][0]
    assert t["sizes"][1:] == [3, 18, 1024], t
    return t["off"] + (t["sizes"][0] - 1) * t["strides"][0] + 2 * t["strides"][1] + 5 * t["strides"][2] + t["sizes"][3] - 4


def timeline(o):
    """Per-op timeline of one dispatch, measured from the host: a thread-free loop polls the first and last word of
    every op's column-0 output in the mapped ACT buffer while the NPU runs and records when each changes."""
    frames = np.load("/work/tapnet/data/clip30.npy")[:3]
    video = preprocess(frames)
    s = Step8(o.build, o.depth)
    q = demo_queries()
    s.reset()
    for t in range(2):  # warm frames (the probed frame is not the first: the SSMs run their recurrent path)
        s.step(video[t], torch.cat([q[..., :1] - t, q[..., 1:]], -1))
    s.select(s.phase)  # round 17: the plan of the phase this probe dispatches (step() leaves s.p at the last dispatched one)
    mv = np.frombuffer(s.bo_a.map(), np.uint32)
    ops = s.p.ops
    marks = [op_marks(op) for op in ops]
    p1m = {i: pass1_mark(op) for i, op in enumerate(ops) if op.get("msplit")}
    only = [x for x in os.environ.get("TL_ONLY", "").split(",") if x]  # round 24: only these op kinds (all blocks)
    sel = [i for i, op in enumerate(ops) if not only or op["name"].split("_", 1)[-1] in only]
    p1m = {i: v for i, v in p1m.items() if i in sel}
    addrs = sorted({a // 4 for i in sel for a in marks[i]} | {a // 4 for a in p1m.values()})
    idx = np.array(addrs)
    # every address is written once per block by the same op kind (shared regions) or once (per-block areas):
    # assign the k-th change of an address to the k-th op (in plan order) that has it as a mark
    users = {a: [] for a in addrs}
    split_last = {}  # round 24: last-word address of split ops -> their instances (assigned by time windows below)
    for i, (f, l) in enumerate(marks):
        if i not in sel:
            continue
        users[f // 4].append((i, 0))
        if i in p1m:  # a split op drains its last tile twice (pass 1 with junk rows 6-17, then pass 2): its last word changes once
            users[p1m[i] // 4].append((i, 2))  # or twice per block (the junk can equal the old value), so it is not counted
            split_last.setdefault(l // 4, []).append(i)
        else:
            users[l // 4].append((i, 1))
    detail = os.environ.get("DETAIL")  # e.g. DETAIL=b5_mlp0: also watch the start of every C object of that op (col 0)
    extra = []
    if detail:
        op = next(op for op in ops if op["name"] == detail)
        if op["kind"] == "ew" and op["fn"] == "ssm":  # the SSM: each head's first output word (C0, rb 0) in columns 0 and 7
            base = op["area"]["base"]
            extra = [base + c * 8 * S + 2 * S + q * KSTEP16 + rb * UNIT for c in (0, 7) for q in range(12) for rb in (0, 17)]
        elif op["kind"] == "attn":  # attention: the first word of every head's O object, column 0, per pass (r7)
            extra = [t["off"] + i * t["strides"][0] + j * t["strides"][1] for pc in op["c"] for t in pc[:1]
                     for i in range(t["sizes"][0]) for j in range(t["sizes"][1])]
        elif op["kind"] == "ew":  # an EW op: the first and last word of every column's output
            extra = [e for c in range(COLS) for e in tap_ends(op["c"][c])]
        else:
            t = op["c"][0]
            f0 = t["off"]
            extra = [f0 + i * t["strides"][0] + j * t["strides"][1] for j in range(t["sizes"][1]) for i in range(t["sizes"][0])]
        addrs = sorted(set(addrs) | {a // 4 for a in extra})
        idx = np.array(addrs)
        users.update({a // 4: users.get(a // 4, []) for a in extra})
    rel = torch.cat([q[..., :1] - 2, q[..., 1:]], -1)
    s.put_x0(video[2], rel)
    for a in addrs:  # (round 24: invalidate before taking the reference values, not only in the poll loop)
        s.bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, 64, (a * 4) // 64 * 64)
    prev = mv[idx].copy()
    changes = {a: [] for a in addrs}
    t0 = time.perf_counter()
    r = s.kern(3, s.bo_i, s.n_instr, s.bo_w, s.bo_a)
    done = pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED
    polls = 0
    while True:
        st = r.state()
        now = time.perf_counter()
        for a in addrs:  # host_only memory is cached on the CPU side: invalidate each probed line
            s.bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, 64, (a * 4) // 64 * 64)
        cur = mv[idx]
        diff = np.nonzero(cur != prev)[0]
        for j in diff:
            changes[addrs[j]].append((now - t0) * 1e3)
        prev = cur.copy()
        polls += 1
        if st == done or now - t0 > 5:
            break
    t_end = (time.perf_counter() - t0) * 1e3
    if detail:
        print(f"object write times of {detail} (column 0, drain order; every change of the word, ms):")
        for a in extra:
            print(f"  +{a - extra[0]:9d} B: " + " ".join(f"{tm:.2f}" for tm in changes[a // 4]))
    print(f"dispatch {t_end:.1f} ms, {polls} polls ({t_end / polls * 1e3:.0f} us each), {len(addrs)} addresses")
    ev = {}  # (op, 0/1/2) -> time
    for a in addrs:
        if users[a] and len(changes[a]) != len(users[a]):  # (a write that left the word unchanged shifts every later op on it)
            print(f"WARNING address +{a * 4} B: {len(changes[a])} changes for {len(users[a])} expected writes "
                  f"({ops[users[a][0][0]]['name']} ..): its marks are unreliable")
        for (key, tm) in zip(users[a], changes[a]):
            ev[key] = tm
    for a, inst in split_last.items():  # instance k ends at its last change between its own pass-1 end and the next instance's
        p1s = [ev.get((i, 2)) for i in inst]
        for k, i in enumerate(inst):
            if p1s[k] is None:
                continue
            hi = next((x for x in p1s[k + 1:] if x is not None), float("inf"))
            ts = [tm for tm in changes[a] if p1s[k] - 1e-6 <= tm < hi - 1e-6]
            if ts:
                ev[(i, 1)] = max(ts)
    rows, last_end = [], 0.0
    for i, op in enumerate(ops):
        f, l = ev.get((i, 0)), ev.get((i, 1))
        kind = op["kind"] if op["kind"] != "ew" else op["fn"]
        rows.append((op["name"], kind, f, l, ev.get((i, 2))))
    print(f"{'op':18s} {'kind':6s} {'first':>8s} {'last':>8s} {'gap':>7s} {'span':>7s}  (ms; gap = first - previous op's last)")
    for name, kind, f, l, p1 in rows:
        fs = f"{f:8.2f}" if f is not None else "       -"
        ls = f"{l:8.2f}" if l is not None else "       -"
        gap = f"{f - last_end:7.2f}" if f is not None else "      -"
        span = f"{l - f:7.2f}" if (f is not None and l is not None) else "      -"
        print(f"{name:18s} {kind:6s} {fs} {ls} {gap} {span}" + (f"  (pass 1 end {p1:.2f})" if p1 is not None else ""))
        if l is not None:
            last_end = l
    out = Path("/work/npu-tapnext/build") / f"tl_{o.build}_d{o.depth}{'_' + detail if detail else ''}.json"
    out.write_text(json.dumps(dict(build=o.build, depth=o.depth, dispatch_ms=t_end, polls=polls,
                                   ops=[dict(name=nm, kind=k, first=f, last=l, pass1=p1) for nm, k, f, l, p1 in rows])))
    print(f"timeline json -> {out.name}")
    # per kind: sum of (last - previous last) = the op's share of the wall time
    share, prev_l = {}, 0.0
    for name, kind, f, l, _ in rows:
        if l is None:
            continue
        k = kind if kind != "gemm" else "gemm:" + name.split("_", 1)[-1]
        share[k] = share.get(k, 0.0) + (l - prev_l)
        prev_l = l
    print("wall-time share by op (last word to last word):")
    for k, v in sorted(share.items(), key=lambda kv: -kv[1]):
        print(f"  {k:16s} {v:8.2f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "blocks", "davis", "time", "timeline", "sweep", "regress", "headscheck"])
    ap.add_argument("--build", required=True)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--videos", type=int, default=10)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--support", type=int, default=0, help="davis: support points per query")
    ap.add_argument("--depths", default="12,10,8,6", help="sweep: depths to run on one xclbin")
    ap.add_argument("--save", default=None, help="regress: save the outputs under this name")
    ap.add_argument("--ref", default=None, help="regress: compare against the outputs saved under this name")
    ap.add_argument("--slots", type=int, default=48, help="davis: query rows (tokens); a build with --attn-keys 1152 for 128")
    o = ap.parse_args()
    if o.depth is None:
        o.depth = int(__import__("re").match(r"full(\d+)", o.build).group(1))
    torch.set_num_threads(int(os.environ.get("THREADS", "8")))
    dict(smoke=smoke, blocks=blocks, davis=davis, time=timing, timeline=timeline, sweep=sweep, regress=regress, headscheck=headscheck)[o.cmd](o)


if __name__ == "__main__":
    main()
