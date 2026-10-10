"""Memory plan + op list for design8 (the fused TAPNext++ step). Host side only: pure python, no IRON.

Every device-side activation lives in one InOut buffer ACT; weights and per-op parameters in W (In). All offsets,
sizes and strides here are BYTES; the design divides by 4 (both buffers are uint32 arrays).

Layouts (per column c of 8; a column owns 144 rows = 18 row-blocks of 8):
  Q16(nq):  [c][q (k-step of 64 cols)][rb 18][UNIT 1600 B = 1024 B bf16 (8 rows x 64 cols, 8x8 blocked) | 576 B bfp panels]
            The GEMM C drain writes the 1024 B parts; a norm op writes X' | panel per unit; a GEMM's A fill
            gathers the 576 B panels at `pan_off` inside the units.
  APAN(nq, kstride): [c][q][kstride B] holding one 10,368 B k-step chunk of A panels ([rb 18][576]) per k-step, written
            by GEMM epilogues as whole 18,432 B C objects (data + junk tail), so kstride >= the window.
  QKV:      [h 12][c 8][12 slots x 4608 B]: slot 3i = Q object (query tile i), 3i+1 = K (key chunk 3c+i), 3i+2 = V^T;
            slots 9..11 take the junk objects of the qkv epilogue.
  SSM area (per block, per column 8 x S): [YX = yx GEMM output Q16(24)][C0][C1][C2][Hlo][Hhi][Z]: 7 input streams S apart
            (y, x, x-1, x-2, x-3, h rows 0-3 fp32, h rows 4-7 fp32) and the output streams x -> C0, C0 -> C1, C1 -> C2,
            h -> Hlo/Hhi, z panels -> Z (panel at offset 0 of the unit).
"""

import json

COLS, M, RB = 8, 144, 18
UNIT, XB, PAN = 1600, 1024, 576
KSTEP16 = RB * UNIT  # one k-step of a Q16 region: 28,800 B
S = 12 * KSTEP16  # a Q16(12) region per column: 345,600 B
CHUNK_A = RB * PAN  # one k-step of A panels: 10,368 B
WIN = 18_432  # a C object (144 x 64 bf16): the drain window of every epilogue
OBJ = 4608  # a B-fifo object / weight tile (64 x 64 bfp16) / attention object slot
QKV_PER_HC = 12 * OBJ  # 55,296 B per (head, column)
QKV_PER_H = COLS * QKV_PER_HC
QKV_BYTES = 12 * QKV_PER_H
O_KSTRIDE = 25_344  # out_proj's A: k-step stride such that both attention passes' windows stay inside the k-step
SLOT = 768  # elementwise ring slot
RA_BYTES = M * 1536 * 9 // 8  # the memtile A slot: 248,832 B
EW_MODE = dict(none=0, rms=1, ln=2, ln_out=3)
EPI = dict(plain=0, gelu=1, gated_a=2, gated_b=3, gelu_pan=4, q_obj=5, k_obj=6, v_obj=7)


def words_a(m, k):
    return m * k * 9 // 8


def bias_obj_flags(epi):
    return EPI[epi]


class Plan:
    def __init__(self):
        self.act_bytes, self.w_bytes = 64, 0
        self.ops = []
        self.regions = {}
        self.writer = {}  # region key -> name of the last gemm/attn op that wrote it (EW ops are synchronous)

    # ---- allocation --------------------------------------------------------------------------------------------
    def alloc(self, nbytes, name=None, align=4096):
        off = (self.act_bytes + align - 1) // align * align
        self.act_bytes = off + nbytes
        if name:
            self.regions[name] = off
        return off

    def alloc_w(self, nbytes):
        off = (self.w_bytes + 63) // 64 * 64
        self.w_bytes = off + nbytes
        return off

    def q16(self, name, nq=12):
        return dict(kind="q16", base=self.alloc(COLS * nq * KSTEP16, name), nq=nq, col=nq * KSTEP16, pan_off=XB)

    def apan(self, name, nq, kstride=WIN):
        return dict(kind="apan", base=self.alloc(COLS * nq * kstride, name), nq=nq, col=nq * kstride, kstride=kstride)

    def qkv(self, name):
        return dict(kind="qkv", base=self.alloc(QKV_BYTES, name))

    def ssm_area(self, name):
        base = self.alloc(COLS * 8 * S, name)
        yx = dict(kind="q16", base=base, nq=24, col=8 * S, pan_off=XB)
        z = dict(kind="q16", base=base + 7 * S, nq=12, col=8 * S, pan_off=0)
        return dict(base=base, yx=yx, z=z)

    # ---- TAP helpers (bytes) ----------------------------------------------------------------------------------
    @staticmethod
    def tap(buf, off, sizes, strides):
        assert len(sizes) == 4 and len(strides) == 4
        # a step over 2^20 words still transfers correctly but is executed per column in turn (measured round 4:
        # norm3 with a 10.5 MB stride took 8 x 0.75 ms, serialised across the columns, instead of 0.8 ms)
        assert all(st // 4 < (1 << 20) for st, n in zip(strides[:3], sizes[:3]) if n > 1), ("shim BD step > 2^20 words", sizes, strides)
        assert all(s % 4 == 0 for s in strides[:3]) and strides[3] == 1 and sizes[3] % 4 == 0 and off % 4 == 0, (off, sizes, strides)
        return dict(buf=buf, off=off, sizes=list(sizes), strides=list(strides))

    def a_fill(self, src, K, c):
        """The A slot load for column c: exactly RA_BYTES (K=768 sources are read twice)."""
        reps = 2 if K == 768 else 1
        if src["kind"] == "q16":
            assert K == 768 and src["nq"] >= 12
            t = self.tap("act", src["base"] + c * src["col"] + src["pan_off"], [reps, 12, RB, PAN], [0, KSTEP16, UNIT, 1])
        elif src["kind"] == "apan":
            nq = K // 64
            assert src["nq"] >= nq
            t = self.tap("act", src["base"] + c * src["col"] + src.get("q0", 0) * src["kstride"], [reps, nq, 1, CHUNK_A], [0, src["kstride"], 0, 1])
        else:
            raise ValueError(src)
        assert reps * (K // 64) * CHUNK_A == RA_BYTES
        return t

    # ---- ops -----------------------------------------------------------------------------------------------------
    def gemm(self, name, src, K, nt, epi="plain", dst=None, flags=None, bias_in=True):
        """A GEMM over the frame: A from `src`, N = 3 * nt * 64 (row r takes N-tiles [r nt, (r+1) nt)), epilogue `epi`.
        `flags`: per-tile epilogue codes (default: all `epi`). The weights go to W at op["w"] (pack8.pack_gemm)."""
        N = 3 * nt * 64
        lw = (K // 64 + 1) * OBJ  # one N-tile: K/64 weight tiles + the bias/flag object
        w = self.alloc_w(3 * nt * lw)
        if flags is None:
            flags = ["gated_a", "gated_b"] * (3 * nt // 2) if epi == "gated" else [epi] * (3 * nt)
        nobj = 3 * nt if epi != "gated" else 3 * nt // 2  # C objects per op (gated: one per tile pair)
        if epi in ("plain", "gelu"):
            if dst is None:
                dst = self.q16(name + "_out", nq=N // 64)
            assert dst["kind"] == "q16" and dst["nq"] >= N // 64
            drains = [self.tap("act", dst["base"] + c * dst["col"], [nt, 3, RB, XB], [KSTEP16, nt * KSTEP16, UNIT, 1]) for c in range(COLS)]
        elif epi in ("gated", "gelu_pan"):
            per_row = nobj // 3
            if dst is None:
                dst = self.apan(name + "_out", nq=nobj)
            assert dst["kind"] == "apan" and dst["kstride"] >= WIN and dst["nq"] >= dst.get("q0", 0) + nobj
            drains = [self.tap("act", dst["base"] + c * dst["col"] + dst.get("q0", 0) * dst["kstride"], [per_row, 3, 1, WIN], [dst["kstride"], per_row * dst["kstride"], 0, 1]) for c in range(COLS)]
        elif epi == "qkv":
            assert nt == 12 and K == 768
            flags = ["q_obj"] * 12 + ["k_obj"] * 12 + ["v_obj"] * 12
            if dst is None:
                dst = self.qkv(name + "_out")
            # a 4608 B object is 1152 words, over the shim's 1023-word inner dim: one drain per tile (12 per column)
            drains = [[self.tap("act", dst["base"] + c * QKV_PER_HC + j * QKV_PER_H, [3, 4, 2, OBJ // 2], [OBJ, 3 * OBJ, OBJ // 2, 1])
                       for j in range(12)] for c in range(COLS)]
        else:
            raise ValueError(epi)
        deps = [self.writer[k] for k in self._keys(src, K=K) if k in self.writer]
        op = dict(kind="gemm", name=name, K=K, nt=nt, N=N, epi=epi, flags=flags, w=w, lw=lw, nsend=(1 if K == 768 else 2) * nt,  # half-slot sends per N-tile: K=768 tiles take one half, K=1536 both
                  a=[self.a_fill(src, K, c) for c in range(COLS)], c=drains, dst=dst, bias_in=bias_in, src=src, deps=sorted(set(deps)))
        self.ops.append(op)
        for k in self._keys(dst, K=None, nq=nobj if epi in ("gated", "gelu_pan") else None):
            self.writer[k] = name
        return dst

    @staticmethod
    def _keys(region, K=None, nq=None):
        """Dependency keys of a region: Q16/QKV -> base; APAN -> (base, 24-k-step window) for the windows touched."""
        if region["kind"] == "apan":
            q0 = region.get("q0", 0)
            n = nq if nq is not None else (K // 64 if K else region["nq"])
            return [(region["base"], w) for w in range(q0 // 24 * 24, q0 + n, 24)]
        return [region["base"]]

    def attn(self, name, qkv, dst=None):
        """Attention on the GEMM cores: Q objects through the A slot (sent once per pass), K/V through the B fifos
        (row r: heads r, r+3, r+6, r+9), O as bfp k-step chunks (head h = k-step h) of out_proj's A."""
        if dst is None:
            dst = self.apan(name + "_out", nq=12, kstride=O_KSTRIDE)
        a = [self.tap("act", qkv["base"] + c * QKV_PER_HC, [2, 12, 3, 3456], [0, QKV_PER_H, 3 * OBJ, 1]) for c in range(COLS)]
        # K_i V_i are adjacent (9216 B = 4 x 2304): one fill per head (4 per row per pass) keeps the inner dim < 1023 words
        b = [[self.tap("act", qkv["base"] + (r + 3 * i) * QKV_PER_H + OBJ, [COLS, 3, 4, OBJ // 2], [QKV_PER_HC, 3 * OBJ, OBJ // 2, 1])
              for i in range(4)] for r in range(3)]
        passes = [(0, 2, 0), (2, 1, 12 * PAN)]  # (first tile, tiles, byte offset inside the k-step)
        c = [[self.tap("act", dst["base"] + col * dst["col"] + off, [4, 3, 1, WIN], [3 * dst["kstride"], dst["kstride"], 0, 1]) for col in range(COLS)]
             for (_, _, off) in passes]
        deps = [self.writer[qkv["base"]]] if qkv["base"] in self.writer else []
        self.ops.append(dict(kind="attn", name=name, a=a, b=b, c=c, passes=passes, nsend=len(passes), dst=dst, deps=deps))
        self.writer[(dst["base"], 0)] = name
        return dst

    def norm(self, name, mode, srcs, params_w, dst=None, params_bytes=None):
        """Elementwise norm op on the row-3 cores: X' = sum(srcs) (bf16); panels = norm(X') or (none) X'; mode ln_out
        writes the LN result into the X' part. `srcs`: 1-3 Q16 regions at equal distances (their X' parts)."""
        srcs = sorted(srcs, key=lambda s: s["base"])  # the sum is commutative: stream order = address order
        n_in = len(srcs)
        if n_in > 1:
            d = srcs[1]["base"] - srcs[0]["base"]
            assert d > 0 and all(srcs[i]["base"] - srcs[0]["base"] == i * d for i in range(n_in)), [s["base"] for s in srcs]
            assert all(s["col"] == srcs[0]["col"] for s in srcs)
        else:
            d = 0
        if dst is None:
            dst = self.q16(name + "_out")
        if params_bytes is None:
            params_bytes = 3 * SLOT if mode in ("rms", "none") else 5 * SLOT  # header slot + vectors
        fills = [[self.tap("w", params_w, [1, 1, 1, params_bytes], [0, 0, 0, 1]),
                  self.tap("act", srcs[0]["base"] + c * srcs[0]["col"], [RB, 12, max(n_in, 1), XB], [UNIT, KSTEP16, d, 1])] for c in range(COLS)]
        drains = [self.tap("act", dst["base"] + c * dst["col"], [RB, 12, 1, UNIT], [UNIT, KSTEP16, 0, 1]) for c in range(COLS)]
        self.ops.append(dict(kind="ew", fn="norm", name=name, mode=mode, n_in=n_in, params_w=params_w, params_bytes=params_bytes,
                             fills=fills, c=drains, dst=dst, srcs=srcs))
        return dst

    def ssm(self, name, area, params_w):
        """The recurrent block's elementwise part on the row-3 cores, head (k-step) major: per head one params fill
        (SSM_PARAM_BYTES) and one chunk fill of the 7 input streams, one drain for the 6 output streams."""
        base = area["base"]
        fills = [[(self.tap("w", params_w + q * SSM_PARAM_BYTES, [1, 1, 1, SSM_PARAM_BYTES], [0, 0, 0, 1]),
                   self.tap("act", base + c * 8 * S + q * KSTEP16, [RB, 7, 1, XB], [UNIT, S, 0, 1])) for q in range(12)] for c in range(COLS)]
        drains = [self.tap("act", base + c * 8 * S + 2 * S, [12, RB, 6, XB], [KSTEP16, UNIT, S, 1]) for c in range(COLS)]
        self.ops.append(dict(kind="ew", fn="ssm", name=name, fills=fills, c=drains, dst=area["z"], area=area, params_w=params_w))
        return area["z"]

    def patch(self, name, frame, params_w, dst=None):
        """Patch embedding (conv 8x8x3 -> 768 as 12 k-step GEMMs on the row-3 cores): frame = region of bf16 patches
        [c][rb][24 col-blocks][8][8] (3072 B per rb); per k-step q: params (PATCH_PARAM_BYTES) + the 18 chunks."""
        if dst is None:
            dst = self.q16(name + "_out")
        fills = [[(self.tap("w", params_w + q * PATCH_PARAM_BYTES, [1, 1, 1, PATCH_PARAM_BYTES], [0, 0, 0, 1]),
                   self.tap("act", frame + c * RB * 3072, [RB, 1, 1, 3072], [3072, 0, 0, 1])) for q in range(12)] for c in range(COLS)]
        drains = [self.tap("act", dst["base"] + c * dst["col"], [12, RB, 1, XB], [KSTEP16, UNIT, 0, 1]) for c in range(COLS)]
        self.ops.append(dict(kind="ew", fn="patch", name=name, fills=fills, c=drains, dst=dst, frame=frame, params_w=params_w))
        return dst

    # ---- serialisation for the design (compile-time) -------------------------------------------------------------
    def sched_json(self):
        keep = ("kind", "fn", "name", "K", "nt", "N", "epi", "flags", "nsend", "w", "a", "b", "c", "fills", "passes", "mode", "n_in", "deps")
        return json.dumps([{k: op[k] for k in keep if k in op} for op in self.ops])


SSM_PARAM_BYTES = 14 * SLOT  # [2 slots of per-head params][Wx 4608][Wa 4608]
PATCH_PARAM_BYTES = 19 * SLOT  # [3 weight tiles 13,824 B][bias 128 B] padded to slots


def block_regions(p: Plan):
    """The working set one block uses (reused by every block); returns a dict of Q16/APAN regions."""
    r = {}
    r["R0"] = p.q16("R0")  # norm0 out: X' = x0, panels = t
    r["R1"] = p.q16("R1")  # linear_out out
    r["D1"] = p.q16("D1")  # ffw_down partials
    r["D2"] = p.q16("D2")
    r["R2"] = p.q16("R2")  # norm1 out: X' = r1 (must follow D2: norm2 reads [D1, D2, R2])
    r["G"] = p.apan("G", nq=48)  # gated GELU k-step chunks (K=3072 A)
    r["R3"] = p.q16("R3")  # norm2 out: X' = x1, panels = l1
    r["R4"] = p.q16("R4")  # out_proj out (must follow R3: norm3 reads [R4, R3] at a stride the shim BD can hold)
    r["QKV"] = p.qkv("QKV")
    r["O"] = p.apan("O", nq=12, kstride=O_KSTRIDE)
    r["M"] = p.apan("M", nq=48)  # mlp0 gelu k-step chunks
    r["D3"] = p.q16("D3")  # mlp3 partials
    r["D4"] = p.q16("D4")
    r["R5"] = p.q16("R5")  # norm3 out: X' = x2, panels = l2 (follows D4: the next norm0 reads [D3, D4, R5])
    return r


def full_step(depth=12, skip=()):
    """The whole frame: `depth` blocks and the encoder norm. The host writes X0 (the X' parts) every frame: the
    patch embedding + pos-emb for the image rows (torch's bf16 conv on the CPU, 0.2 ms) and the query tokens.
    Returns (plan, info) where info names the regions the host touches (X0, SSM state areas, the final LN rows)."""
    p = Plan()
    r = block_regions(p)
    x0 = p.q16("X0")
    areas = [p.ssm_area(f"SSM{b}") for b in range(depth)]
    fin = p.q16("FINAL")
    w = {}
    prev = [x0]  # inputs of block 0's norm0
    for b in range(depth):
        wb = w[b] = {}
        wb["norm0"] = p.alloc_w(3 * SLOT)
        p.norm(f"b{b}_norm0", "rms", prev, wb["norm0"], dst=r["R0"])
        p.gemm(f"b{b}_yx", r["R0"], 768, 8, epi="gelu", flags=["gelu"] * 12 + ["plain"] * 12, dst=areas[b]["yx"])
        wb["ssm"] = p.alloc_w(12 * SSM_PARAM_BYTES)
        if "ssm" not in skip:
            p.ssm(f"b{b}_ssm", areas[b], wb["ssm"])
        p.gemm(f"b{b}_out", areas[b]["z"], 768, 4, dst=r["R1"])
        wb["norm1"] = p.alloc_w(3 * SLOT)
        p.norm(f"b{b}_norm1", "rms", [r["R1"], r["R0"]], wb["norm1"], dst=r["R2"])
        p.gemm(f"b{b}_up_a", r["R2"], 768, 16, epi="gated", dst=dict(r["G"], q0=0))
        p.gemm(f"b{b}_up_b", r["R2"], 768, 16, epi="gated", dst=dict(r["G"], q0=24))
        p.gemm(f"b{b}_down_a", dict(r["G"], q0=0), 1536, 4, dst=r["D1"])
        p.gemm(f"b{b}_down_b", dict(r["G"], q0=24), 1536, 4, dst=r["D2"], bias_in=False)
        wb["norm2"] = p.alloc_w(5 * SLOT)
        p.norm(f"b{b}_norm2", "ln", [r["D1"], r["D2"], r["R2"]], wb["norm2"], dst=r["R3"])
        p.gemm(f"b{b}_qkv", r["R3"], 768, 12, epi="qkv", dst=r["QKV"])
        if "attn" not in skip:
            p.attn(f"b{b}_attn", r["QKV"], dst=r["O"])
        p.gemm(f"b{b}_out_proj", r["O"], 768, 4, dst=r["R4"])
        wb["norm3"] = p.alloc_w(5 * SLOT)
        p.norm(f"b{b}_norm3", "ln", [r["R4"], r["R3"]], wb["norm3"], dst=r["R5"])
        p.gemm(f"b{b}_mlp0", r["R5"], 768, 16, epi="gelu_pan", dst=dict(r["M"], q0=0))
        p.gemm(f"b{b}_mlp3_a", dict(r["M"], q0=0), 1536, 4, dst=r["D3"])
        p.gemm(f"b{b}_mlp3_b", dict(r["M"], q0=24), 1536, 4, dst=r["D4"], bias_in=False)
        prev = [r["D3"], r["D4"], r["R5"]]
    w["final"] = p.alloc_w(5 * SLOT)
    p.norm("final", "ln_out", prev, w["final"], dst=fin)
    info = dict(x0=x0, areas=areas, final=fin, w=w, regions=r)
    return p, info


if __name__ == "__main__":
    p, info = full_step()
    print(f"ACT {p.act_bytes / 1e6:.1f} MB, W {p.w_bytes / 1e6:.1f} MB, ops {len(p.ops)}")
    print(f"sched json {len(p.sched_json()) / 1e3:.0f} KB")
