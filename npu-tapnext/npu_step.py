"""TAPNext++ one-frame step with the trunk GEMMs and the attention executed ON THE NPU (scratch prototype).

Runs inside the npu-iron container (pyxrt + aie.utils.bfp + torch CPU). The vendored model
(agent/app/jobs/tapnext/model.py) runs on the CPU in bf16; F.linear of every trunk Linear is routed to the
NPU for the 1024 image rows (bfp16ebs8 operands, fp32 accumulation; a K=3072 Linear runs as four K=768 chunks
whose fp32 partials are summed on the host) and to the CPU (bf16) for the 48 query rows; F.scaled_dot_product_
attention is routed to the NPU attention program for all 1072 tokens. One xclbin per GEMM shape (5) + the
attention xclbin: every op is its own dispatch, so the step time here is dominated by hardware-context switches
and host packing, NOT by the kernels - the per-op kernel times are what transfers to a fused program.

usage (in the container, repo mounted at /work):
  python3 npu_step.py smoke            one frame vs the all-CPU bf16 step: per-block error, tracks
  python3 npu_step.py davis [--videos 10]   TAP-Vid DAVIS 'first' AJ with the NPU numerics
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from ml_dtypes import bfloat16 as mlbf16

sys.path.insert(0, "/opt/xilinx/xrt/python")
import pyxrt  # noqa: E402

ROOT = "/work/.scratch/npu-tapnext"
sys.path.insert(0, ROOT)
from pack import pack_a, pack_w  # noqa: E402  (bfp panel orders of the GEMM program)

COLS, ROWS, M_CORE, KT, NT = 8, 4, 128, 64, 64   # the single-op GEMM programs: 1024 rows, k 64, n 64
RHO = 8
M_IMG = COLS * M_CORE
import types
tn = types.ModuleType("tn")  # the vendored model targets Python 3.14 (lazy annotations); the container is 3.13
exec(open("/work/agent/app/jobs/tapnext/model.py").read().replace("-> TAPNext:", '-> "TAPNext":'), tn.__dict__)

STATS = dict(npu_us=0.0, pack_ms=0.0, gemm_runs=0, attn_runs=0, cpu_rows_ms=0.0)


# ---------------------------------------------------------------- fast bfp16ebs8 encoder (torch) ----------
def encode_rows_fast(x: torch.Tensor) -> np.ndarray:
    """float (R, K) -> uint8 (R, K/8, 9) with the mlir-aie codec (conv_even). A torch re-implementation was 2x
    faster but differed on 0.25% of the blocks (its mantissa-128 bump rule): the codec is what the kernels were
    verified against, so it stays."""
    from aie.utils import bfp
    R, K = x.shape
    return bfp.encode(np.ascontiguousarray(x.detach().float().numpy()), rounding="conv_even").reshape(R, K // 8, 9)


def pack_act(x_rows: torch.Tensor) -> np.ndarray:
    """[1024, K] activations -> the program's A words (row-half-major, -DBLK12)."""
    t0 = time.perf_counter()
    words = pack_a(encode_rows_fast(x_rows), M_CORE, KT, RHO, half=True)
    STATS["pack_ms"] += (time.perf_counter() - t0) * 1e3
    return words


def unblock_c(cbuf: np.ndarray, N: int) -> np.ndarray:
    """C words of one op (c_linear: per-core tiles [m/8][n/8][8][8]) -> [1024, N] float32."""
    c = cbuf.reshape(COLS * M_CORE, N)
    tiles = c.reshape(COLS, M_CORE, N // NT, NT).transpose(0, 2, 1, 3)
    tiles = tiles.reshape(COLS, N // NT, M_CORE // 8, NT // 8, 8, 8)
    return tiles.transpose(0, 2, 4, 1, 3, 5).reshape(COLS * M_CORE, N).astype(np.float32)


# ---------------------------------------------------------------- NPU programs ------------------------------
class GemmProgram:
    """One single-op xclbin (K 768, nt N tiles per core -> N = 256 nt)."""

    def __init__(self, dev, nt):
        self.nt, self.N = nt, ROWS * nt * NT
        b = f"{ROOT}/build/op_nt{nt}"
        self.xb = pyxrt.xclbin(b + ".xclbin"); dev.register_xclbin(self.xb)
        self.ctx = pyxrt.hw_context(dev, self.xb.get_uuid())
        self.kern = pyxrt.kernel(self.ctx, self.xb.get_kernels()[0].get_name())
        instr = np.fromfile(b + ".bin", dtype=np.uint32)
        self.bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, self.kern.group_id(1))
        self.bo_i.write(instr.tobytes(), 0); self.bo_i.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self.n_instr = len(instr)
        self.la = M_CORE * 768 * 9 // 8 // 4                       # words of one core's A (128 x 768)
        self.bo_a = pyxrt.bo(dev, COLS * self.la * 4, pyxrt.bo.host_only, self.kern.group_id(3))
        self.c_words = COLS * M_CORE * self.N                        # bf16 elements
        self.bo_c = pyxrt.bo(dev, self.c_words * 2, pyxrt.bo.host_only, self.kern.group_id(4))
        self.dev = dev

    def weight_bo(self, w_rows: torch.Tensor):
        """W^T [N, 768] (torch, any float dtype) -> a device buffer with the program's W layout."""
        ew = encode_rows_fast(w_rows)
        words = pack_w(ew, NT, KT)
        bo = pyxrt.bo(self.dev, words.nbytes, pyxrt.bo.host_only, self.kern.group_id(3))
        bo.write(words.tobytes(), 0); bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        return bo

    def run(self, a_words: np.ndarray, bo_w) -> np.ndarray:
        self.bo_a.write(a_words.tobytes(), 0); self.bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        t0 = time.perf_counter()
        r = self.kern(3, self.bo_i, self.n_instr, self.bo_a, bo_w, self.bo_c)
        st = r.wait(3_000_000)
        STATS["npu_us"] += (time.perf_counter() - t0) * 1e6; STATS["gemm_runs"] += 1
        assert str(st) == "ert_cmd_state.ERT_CMD_STATE_COMPLETED", st
        self.bo_c.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        c = np.frombuffer(self.bo_c.read(self.c_words * 2, 0), dtype=mlbf16)
        return unblock_c(c, self.N)


class AttnProgram:
    """attn_tn.xclbin: 12 heads x 1072 tokens x 64 dims; Q/K/V^T as bfp16 panel objects, one stream per column."""

    def __init__(self, dev):
        b = f"{ROOT}/build/attn_tn"
        cfg = json.load(open(b + ".json"))
        self.QT, self.HD, self.KC, self.NK, self.ITEMS = cfg["QT"], cfg["HD"], cfg["KC"], cfg["NKEYS"], cfg["ITEMS"]
        self.H, self.QTILES = cfg["HEADS"], cfg["QTILES"]
        self.NCH = -(-self.NK // self.KC); self.NQT = -(-self.QTILES // 4) * 4; self.PASSES = self.NQT // 4
        self.GQ, self.GV, self.OBJB = 576, 448, 3584
        self.xb = pyxrt.xclbin(b + ".xclbin"); dev.register_xclbin(self.xb)
        self.ctx = pyxrt.hw_context(dev, self.xb.get_uuid())
        self.kern = pyxrt.kernel(self.ctx, self.xb.get_kernels()[0].get_name())
        instr = np.fromfile(b + ".bin", dtype=np.uint32)
        self.bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, self.kern.group_id(1))
        self.bo_i.write(instr.tobytes(), 0); self.bo_i.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self.n_instr = len(instr)
        self.obj = self.OBJB // 4
        self.per_item = ROWS + 2 * self.NCH
        self.n_kq = COLS * self.ITEMS * self.per_item * self.obj
        self.n_o = COLS * ROWS * self.ITEMS * self.QT * self.HD
        self.bo_kq = pyxrt.bo(dev, self.n_kq * 4, pyxrt.bo.host_only, self.kern.group_id(3))
        self.bo_o = pyxrt.bo(dev, self.n_o * 2, pyxrt.bo.host_only, self.kern.group_id(4))

    @staticmethod
    def _blocked(m):  # (rows, cols) -> [rb][cb][8][8] flattened
        r, c = m.shape
        return m.reshape(r // 8, 8, c // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8)

    def _groups(self, parts, gbytes):
        out = np.zeros(self.OBJB, np.uint8); off = 0
        for p in parts:
            out[off:off + len(p)] = p; off += gbytes
        return out.view(np.uint32)

    def _qk_obj(self, t):  # 48 x 64 rows x dims -> 6 groups of 8 panels
        return self._groups([encode_rows_fast(torch.from_numpy(np.ascontiguousarray(self._blocked(t[rb * 8:(rb + 1) * 8])))).reshape(-1) for rb in range(6)], self.GQ)

    def _v_obj(self, v):  # 48 keys x 64 dims -> 8 groups of 6 V^T panels [8 dims][8 keys]
        return self._groups([encode_rows_fast(torch.from_numpy(np.ascontiguousarray(np.concatenate([v[cb * 8:(cb + 1) * 8, j * 8:(j + 1) * 8].T.reshape(-1, 8) for cb in range(6)])))).reshape(-1) for j in range(8)], self.GV)

    def run(self, q, k, v):
        """q, k, v: [H, NK, HD] float32 -> o [H, NK, HD] float32 (bf16 values)."""
        t0 = time.perf_counter()
        H, NK, HD, QT, KC = self.H, self.NK, self.HD, self.QT, self.KC
        Kp = np.zeros((H, self.NCH * KC, HD), np.float32); Kp[:, :NK] = k
        Vp = np.zeros((H, self.NCH * KC, HD), np.float32); Vp[:, :NK] = v
        Qp = np.zeros((H, self.NQT * QT, HD), np.float32); Qp[:, :NK] = q
        kv_cache = {}
        KQ = np.zeros((COLS, self.ITEMS * self.per_item, self.obj), np.uint32)
        for c in range(COLS):
            stream = []
            for i in range(self.ITEMS):
                g = c + COLS * i; h, p = g // self.PASSES, g % self.PASSES
                for r in range(ROWS):
                    qt = 4 * p + r
                    stream.append(self._qk_obj(Qp[h, qt * QT:(qt + 1) * QT]))
                if h not in kv_cache:
                    kv_cache[h] = [(self._qk_obj(Kp[h, j * KC:(j + 1) * KC]), self._v_obj(Vp[h, j * KC:(j + 1) * KC])) for j in range(self.NCH)]
                for kk, vv in kv_cache[h]:
                    stream += [kk, vv]
            KQ[c] = np.stack(stream)
        STATS["pack_ms"] += (time.perf_counter() - t0) * 1e3
        self.bo_kq.write(KQ.tobytes(), 0); self.bo_kq.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        t1 = time.perf_counter()
        r = self.kern(3, self.bo_i, self.n_instr, self.bo_kq, self.bo_o)
        st = r.wait(3_000_000)
        STATS["npu_us"] += (time.perf_counter() - t1) * 1e6; STATS["attn_runs"] += 1
        assert str(st) == "ert_cmd_state.ERT_CMD_STATE_COMPLETED", st
        self.bo_o.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        O = np.frombuffer(self.bo_o.read(self.n_o * 2, 0), dtype=mlbf16).astype(np.float32).reshape(COLS, self.ITEMS, ROWS, QT * HD)
        out = np.zeros((H, self.NQT * QT, HD), np.float32)
        for c in range(COLS):
            for i in range(self.ITEMS):
                g = c + COLS * i; h, p = g // self.PASSES, g % self.PASSES
                for rr in range(ROWS):
                    qt = 4 * p + rr
                    out[h, qt * QT:(qt + 1) * QT] = O[c, i, rr].reshape(QT // 8, HD // 8, 8, 8).transpose(0, 2, 1, 3).reshape(QT, HD)
        return out[:, :NK]


# ---------------------------------------------------------------- the model with NPU routing ---------------
class NpuTapNext:
    def __init__(self):
        self.dev = pyxrt.device(0)
        self.gemms = {nt: GemmProgram(self.dev, nt) for nt in (3, 6, 9, 12, 24)}
        self.attn = AttnProgram(self.dev)
        self.model = tn.TAPNext()
        tn.load_state(self.model, "/work/data/models/tapnextpp_256.safetensors")
        self.model.to_half(torch.bfloat16).eval()
        # every trunk Linear: (nt per K chunk, [weight bo per chunk], bias)
        self.ops = {}
        t0 = time.perf_counter()
        for i, blk in enumerate(self.model.blocks):
            for lin in (blk.ssm_block.recurrent_block.linear_yx, blk.ssm_block.recurrent_block.linear_out,
                        blk.ssm_block.mlp_block.ffw_up, blk.ssm_block.mlp_block.ffw_down,
                        blk.vit_block.self_attention.out_proj, blk.vit_block.mlp[0], blk.vit_block.mlp[3]):
                self._register(lin.weight)
            self._register(blk.vit_block.self_attention.in_proj_weight)
        print(f"packed {len(self.ops)} trunk Linears to the NPU in {time.perf_counter() - t0:.0f}s", flush=True)
        self._orig_linear, self._orig_sdpa = F.linear, F.scaled_dot_product_attention

    def _register(self, w):
        N, K = w.shape
        assert K % 768 == 0 and N % 256 == 0, (N, K)
        nt = N // 256
        prog = self.gemms[nt]
        bos = [prog.weight_bo(w[:, c * 768:(c + 1) * 768].float()) for c in range(K // 768)]
        self.ops[id(w)] = (nt, bos)

    def linear(self, x, w, b=None):
        op = self.ops.get(id(w))
        if op is None or x.dim() != 2 or x.shape[0] != M_IMG + 48:
            return self._orig_linear(x, w, b)
        nt, bos = op
        prog = self.gemms[nt]
        acc = np.zeros((M_IMG, prog.N), np.float32)
        for c, bo in enumerate(bos):
            acc += prog.run(pack_act(x[:M_IMG, c * 768:(c + 1) * 768]), bo)
        y_img = torch.from_numpy(acc)
        t0 = time.perf_counter()
        y_q = self._orig_linear(x[M_IMG:], w, None).float()   # the 48 query rows on the CPU (bf16)
        STATS["cpu_rows_ms"] += (time.perf_counter() - t0) * 1e3
        y = torch.cat([y_img, y_q])
        if b is not None:
            y = y + b.float()
        return y.to(x.dtype)

    def sdpa(self, q, k, v, **kw):
        if q.dim() != 4 or q.shape[2] != M_IMG + 48:
            return self._orig_sdpa(q, k, v, **kw)
        o = self.attn.run(q[0].float().numpy(), k[0].float().numpy(), v[0].float().numpy())
        return torch.from_numpy(o)[None].to(q.dtype)

    def __enter__(self):
        F.linear = self.linear; F.scaled_dot_product_attention = self.sdpa; return self

    def __exit__(self, *a):
        F.linear, F.scaled_dot_product_attention = self._orig_linear, self._orig_sdpa


def preprocess(frames_u8, res=256):
    t = torch.from_numpy(np.ascontiguousarray(frames_u8)).permute(0, 3, 1, 2).float()
    t = F.interpolate(t, size=(res, res), mode="bilinear", align_corners=False)
    return (t / 127.5 - 1.0).permute(0, 2, 3, 1).contiguous()


def smoke():
    sys.path.insert(0, "/work/.scratch/tapnet")
    frames = np.load("/work/.scratch/tapnet/data/clip30.npy")[:3]
    video = preprocess(frames)
    net = NpuTapNext()
    q = torch.tensor([[[0.0, 100.0, 100.0], [0.0, 200.0, 60.0], [0.0, 50.0, 220.0], [0.0, 128.0, 128.0]] + [[10000.0, 0.0, 0.0]] * 44])
    with torch.no_grad():
        ref = []; caches = None
        for t in range(3):
            rel = torch.cat([q[..., :1] - t, q[..., 1:]], -1)
            tr, lg, vs, caches = net.model.step(video[t], rel, caches)
            ref.append((tr.clone(), vs.clone()))
        for k in STATS: STATS[k] = 0
        caches = None; got = []
        with net:
            for t in range(3):
                t0 = time.perf_counter()
                rel = torch.cat([q[..., :1] - t, q[..., 1:]], -1)
                tr, lg, vs, caches = net.model.step(video[t], rel, caches)
                print(f"frame {t}: step {time.perf_counter() - t0:.2f}s, NPU kernels {STATS['npu_us']/1e3:.1f} ms ({STATS['gemm_runs']} gemm + {STATS['attn_runs']} attn dispatches), packing {STATS['pack_ms']:.0f} ms, cpu query rows {STATS['cpu_rows_ms']:.0f} ms", flush=True)
                for k in STATS: STATS[k] = 0
                got.append((tr.clone(), vs.clone()))
    for t in range(3):
        print(f"frame {t} tracks (y,x) CPU bf16: {ref[t][0][:4].numpy().round(2).tolist()}")
        print(f"frame {t} tracks (y,x) NPU     : {got[t][0][:4].numpy().round(2).tolist()}  max |d| = {(ref[t][0][:4]-got[t][0][:4]).abs().max():.2f} px; vis CPU {ref[t][1][:4].numpy().round(2).tolist()} NPU {got[t][1][:4].numpy().round(2).tolist()}")


def davis(videos):
    # davis.py imports tnpp (the repo model: einops/torchvision); only its loader, query sampling and metrics are needed
    src = "\n".join(l for l in open("/work/.scratch/tapnet/davis.py").read().split("\n") if "import tnpp" not in l)
    dv = types.ModuleType("dv"); dv.__dict__["__file__"] = "/work/.scratch/tapnet/davis.py"
    exec(src.split("def track_video")[0], dv.__dict__)
    dv.ROOT = __import__("pathlib").Path("/work/.scratch/tapnet")
    data = dv.load_davis(dv.ROOT / "data" / "tapvid_davis.zip")
    names = sorted(data.keys())[:videos]
    net = NpuTapNext()
    per_video, ms = [], []
    for name in names:
        d = data[name]; frames = d["video"]; T = frames.shape[0]
        pts = d["points"] * np.array([256.0, 256.0])
        qp, gt_tracks, gt_occ = dv.sample_queries_first(d["occluded"], pts)
        Q = qp.shape[1]
        q = torch.from_numpy(qp.astype(np.float32))                # [1,Q,3]
        q = torch.cat([q, torch.full((1, 48 - Q, 3), 10000.0)], 1) if Q < 48 else q[:, :48]
        video = preprocess(frames)
        caches = None; tr, vl = [], []
        for k in STATS: STATS[k] = 0
        t1 = time.perf_counter()
        with torch.no_grad(), net:
            for t in range(T):
                rel = torch.cat([q[..., :1] - t, q[..., 1:]], -1)
                tracks, logits, vis, caches = net.model.step(video[t], rel, caches)
                tr.append(tracks[:Q].clone()); vl.append(vis[:Q].clone())
        ms.append((time.perf_counter() - t1) * 1e3 / T)
        tracks = torch.stack(tr).numpy(); vis = torch.stack(vl).numpy()
        pred_tracks = tracks[..., ::-1].transpose(1, 0, 2)[None]; pred_occ = ~(vis > 0).T[None]
        m = dv.compute_tapvid_metrics(qp, gt_occ, gt_tracks, pred_occ, pred_tracks)
        per_video.append({k: float(np.mean(v)) for k, v in m.items()})
        print(f"{name:24s} T={T:3d} Q={Q:3d} AJ={per_video[-1]['average_jaccard']*100:5.1f} OA={per_video[-1]['occlusion_accuracy']*100:5.1f} PTS={per_video[-1]['average_pts_within_thresh']*100:5.1f} {ms[-1]:.0f} ms/f (NPU kernels {STATS['npu_us']/1e3/T:.1f} ms/f, packing {STATS['pack_ms']/T:.0f} ms/f)", flush=True)
    agg = {k: round(float(np.mean([v[k] for v in per_video])) * 100, 2) for k in per_video[0]}
    row = dict(mode="npu_real", videos=len(names), ms_per_frame=round(float(np.median(ms)), 1), **agg)
    with open(f"{ROOT}/davis_npu.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print("RESULT", json.dumps(row)); print("DAVIS_DONE")


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", "8")))
    if sys.argv[1] == "smoke":
        smoke()
    else:
        davis(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
