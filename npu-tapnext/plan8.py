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
import os

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
# round 10 (research P5a): A_HALF=1 loads each half of the memtile A slot through its own memtile S2MM channel, both fed by
# one multicast stream. A K=768 op's A (and attention's Q objects) leaves DDR once (124 KB instead of 248); a K=1536 op
# streams its 248 KB as before; an op with reuse_a (up_b: the same A as up_a) loads nothing. Fills are lists.
A_HALF = os.environ.get("A_HALF") == "1"
# round 10 (research P23): P23=1 deals the N tiles of ops built with rr=True (yx, qkv) round-robin to the 3 GEMM rows (row r,
# step j -> natural tile 3 j + r) instead of contiguous ranges, so the rows of a column (coupled by the A broadcast) run the
# same epilogue kind at the same time. Only the weight / flag order (pack8) and the drain TAPs change.
P23 = os.environ.get("P23") == "1"
# round 20 (IL): mlp0 dealt round-robin (step j of row r = chunk 3 j + r: chunks 0-23 are out after step 7), its drain split at
# step 8 (group "<name>#0" = steps 0-7 = mlp3_a's whole A), and mlp3_a's A loaded half 1 first (chunks 12-23 into the half mlp0
# leaves free, while mlp0 still runs), then half 0 as mlp0's last sends free it (design8: per-half full locks).
IL = os.environ.get("IL") == "1"
DAPRE = os.environ.get("DAPRE") == "1"
# round 20 (research rev 16 NRES): the named norms (e.g. "norm1,norm3") send their panels from the EW core's MM2S 1 straight into
# memtile c's A slot half 0 (rb-major: unit (rb, q) at rb 6,912 + q 576); only X' goes to DDR; the GEMM after the norm loads no A
NRES = tuple(x for x in os.environ.get("NRES", "").split(",") if x)  # round 20 (research rev 15 R4 / DAPRE): down_a's A half 1 first too (it loads during up_b)
# round 15 (research rev 10 P60g): P60G=1 deals the attention's query half tiles (3 row blocks, 1,728 B per head) across the
# columns in the two GATHERS only (the attention Q fill and out_proj's A fill; drains, K / V, cores, host unchanged), so the
# lazy-max slow rows (clustered in the lower image) spread over the columns that the K / V multicast keeps in lockstep.
# Group G = {0, 1, 3, 4, 6, 7}: dealt column 3T + e takes half e of tile T from natural columns 3j + H at slot position 2j + H;
# pair {2, 5}: dealt 2 the half-0 halves of natural 2 (positions 0-2) and 5 (3-5), dealt 5 the half-1 halves (r15/deal4.py).
P60G = os.environ.get("P60G") == "1"
HQ = 3 * PAN  # a half tile per head: 1,728 B
# round 17 (X1 full rotation, research rev 12 design; replaces the round-17 sliding area): XROT=4 keeps the conv history in four
# regions R0..R3 (where x, C0, C1, C2 sat: B + S .. B + 4 S) and rotates roles instead of copying: frame t (phase r = t mod 4) has
# the yx GEMM write x(t) into R[r] (its drain splits y | x), the SSM fill stays ONE TAP over [y, R0..R3, Hlo, Hhi], the kernel maps
# tap x - k to R[(r - k) mod 4] (params header word 2 = 1 + r: four params copies in W), and the drain writes only [h lo, h hi, z]
# at B + 5 S (h over the Hlo / Hhi this chunk read first). Four runtime sequences per depth (XPH = the phase this plan builds);
# the layout and the memory stay as they were.
XROT = int(os.environ.get("XROT", "0") or 0)
# round 17 (research rev 12 AL1): every EW op's params fill padded to a multiple of 3 KB = lcm(1 KB, the 768 B ring slot: the core's
# packet BD fills whole slots, so a fill group must stay a slot multiple - 1 KB padding hung the step) (SSM 10,752 -> 12,288 B per
# head, RMS 2,304 -> 3,072, LN 3,840 -> 6,144): the input ring pointer stays 1 KB aligned, no 1 KB SSM part straddles the ring end
AL1 = os.environ.get("AL1") == "1"


def kb_pad(n):
    return (n + 3071) // 3072 * 3072 if AL1 else n
XPH = int(os.environ.get("XPH", "0") or 0)
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

    def ssm_area(self, name, phase=0):
        base = self.alloc(COLS * 8 * S, name)
        yx = dict(kind="q16", base=base, nq=24, col=8 * S, pan_off=XB)
        z = dict(kind="q16", base=base + 7 * S, nq=12, col=8 * S, pan_off=0)
        if not XROT:
            return dict(base=base, yx=yx, z=z, col=8 * S, out=base + 2 * S, rot=None, span=COLS * 8 * S)
        assert XROT == 4 and 0 <= phase < 4
        yx["x_at"] = base + (1 + phase) * S  # x(t) -> R[t mod 4]
        return dict(base=base, yx=yx, z=z, col=8 * S, out=base + 5 * S, rot=phase, span=COLS * 8 * S)

    # ---- TAP helpers (bytes) ----------------------------------------------------------------------------------
    @staticmethod
    def tap(buf, off, sizes, strides):
        assert len(sizes) == 4 and len(strides) == 4
        # a step over 2^20 words still transfers correctly but is executed per column in turn (measured round 4:
        # norm3 with a 10.5 MB stride took 8 x 0.75 ms, serialised across the columns, instead of 0.8 ms)
        assert all(st // 4 < (1 << 20) for st, n in zip(strides[:3], sizes[:3]) if n > 1), ("shim BD step > 2^20 words", sizes, strides)
        assert all(s % 4 == 0 for s in strides[:3]) and strides[3] == 1 and sizes[3] % 4 == 0 and off % 4 == 0, (off, sizes, strides)
        return dict(buf=buf, off=off, sizes=list(sizes), strides=list(strides))

    def a_fill(self, src, K, c, deal=False):
        """The A slot load for column c: exactly RA_BYTES (K=768 sources are read twice). A_HALF: a list of fills, each
        a half slot (RA_BYTES / 2) tagged with the halves it lands in."""
        if A_HALF:
            nq = K // 64
            if src["kind"] == "q16":
                assert K == 768 and src["nq"] >= 12
                return [dict(self.tap("act", src["base"] + c * src["col"] + src["pan_off"], [1, 12, RB, PAN], [0, KSTEP16, UNIT, 1]), pkt="b")]
            assert src["kind"] == "apan" and src["nq"] >= nq, src
            base = src["base"] + c * src["col"] + src.get("q0", 0) * src["kstride"]
            if deal:  # P60G: natural column c gathers its half tiles from the dealt columns' O (pieces in natural order)
                assert K == 768 and src.get("q0", 0) == 0
                O, col, ks = src["base"], src["col"], src["kstride"]
                if c in (2, 5):
                    t = self.tap("act", O + 2 * col + 3 * (c == 5) * HQ, [12, 3, 2, HQ], [ks, HQ, 3 * col, 1])
                else:
                    j, H = divmod(c, 3)
                    t = self.tap("act", O + (2 * j + H) * HQ, [12, 3, 2, HQ], [ks, 3 * col, col, 1])
                return [dict(t, pkt="b")]
            if K == 768:
                return [dict(self.tap("act", base, [1, 12, 1, CHUNK_A], [0, src["kstride"], 0, 1]), pkt="b")]
            assert K == 1536  # the whole 248 KB: design8 writes it over the slot (channel 0) and twice over half 1 (channel 5)
            return [dict(self.tap("act", base, [1, 24, 1, CHUNK_A], [0, src["kstride"], 0, 1]), pkt="b")]
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
    def gemm(self, name, src, K, nt, epi="plain", dst=None, flags=None, bias_in=True, reuse_a=False, rr=False, deal=False, split=0,
             h1first=False, nres=False):
        """A GEMM over the frame: A from `src`, N = 3 * nt * 64 (row r takes N-tiles [r nt, (r+1) nt)), epilogue `epi`.
        `flags`: per-tile epilogue codes (default: all `epi`). The weights go to W at op["w"] (pack8.pack_gemm).
        `reuse_a` (A_HALF only): the A slot still holds this op's A (the previous slot op read the same K=768 region): no load."""
        N = 3 * nt * 64
        rr = rr and P23
        assert not rr or epi in ("plain", "gelu", "qkv", "gelu_pan"), name
        lw = (K // 64 + 1) * OBJ  # one N-tile: K/64 weight tiles + the bias/flag object
        w = self.alloc_w(3 * nt * lw)
        if flags is None:
            flags = ["gated_a", "gated_b"] * (3 * nt // 2) if epi == "gated" else [epi] * (3 * nt)
        nobj = 3 * nt if epi != "gated" else 3 * nt // 2  # C objects per op (gated: one per tile pair)
        if epi in ("plain", "gelu"):
            if dst is None:
                dst = self.q16(name + "_out", nq=N // 64)
            assert dst["kind"] == "q16" and dst["nq"] >= N // 64
            st_j, st_r = (3 * KSTEP16, KSTEP16) if rr else (KSTEP16, nt * KSTEP16)  # output k-step of (step j, row r)
            drains = [self.tap("act", dst["base"] + c * dst["col"], [nt, 3, RB, XB], [st_j, st_r, UNIT, 1]) for c in range(COLS)]
            if "x_at" in dst:  # round 17 XROT: the yx output in k-step order (rr), y (k-steps 0-11) at base, x (12-23) at x_at
                assert rr and nt == 8
                drains = [self.tap("act", dst["base"] + c * dst["col"], [2, 12, RB, XB], [dst["x_at"] - dst["base"], KSTEP16, UNIT, 1])
                          for c in range(COLS)]
        elif epi in ("gated", "gelu_pan"):
            per_row = nobj // 3
            if dst is None:
                dst = self.apan(name + "_out", nq=nobj)
            assert dst["kind"] == "apan" and dst["kstride"] >= WIN and dst["nq"] >= dst.get("q0", 0) + nobj
            st_j, st_r = (3 * dst["kstride"], dst["kstride"]) if rr else (dst["kstride"], per_row * dst["kstride"])  # round 20 IL: rr
            drains = [self.tap("act", dst["base"] + c * dst["col"] + dst.get("q0", 0) * dst["kstride"], [per_row, 3, 1, WIN], [st_j, st_r, 0, 1]) for c in range(COLS)]
        elif epi == "qkv":
            assert nt == 12 and K == 768
            flags = ["q_obj"] * 12 + ["k_obj"] * 12 + ["v_obj"] * 12
            if dst is None:
                dst = self.qkv(name + "_out")
            # a 4608 B object is 1152 words, over the shim's 1023-word inner dim: one drain per tile (12 per column)
            if rr:  # step j: all rows run kind j // 4 (Q, K, V) for heads 3 (j % 4) + r
                drains = [[self.tap("act", dst["base"] + c * QKV_PER_HC + 3 * (j % 4) * QKV_PER_H + (j // 4) * OBJ, [3, 4, 2, OBJ // 2],
                                    [QKV_PER_H, 3 * OBJ, OBJ // 2, 1]) for j in range(12)] for c in range(COLS)]
            else:
                drains = [[self.tap("act", dst["base"] + c * QKV_PER_HC + j * QKV_PER_H, [3, 4, 2, OBJ // 2], [OBJ, 3 * OBJ, OBJ // 2, 1])
                           for j in range(12)] for c in range(COLS)]
        else:
            raise ValueError(epi)
        deps = [self.writer[k] for k in self._keys(src, K=K) if k in self.writer]
        op = dict(kind="gemm", name=name, K=K, nt=nt, N=N, epi=epi, flags=flags, w=w, lw=lw, nsend=(1 if K == 768 else 2) * nt,  # half-slot sends per N-tile: K=768 tiles take one half, K=1536 both
                  a=[self.a_fill(src, K, c, deal) for c in range(COLS)], c=drains, dst=dst, bias_in=bias_in, src=src, deps=sorted(set(deps)))
        if rr:
            op["rr"] = True
        if split:  # round 20 IL: design8 drains steps [0, split) in group "<name>#0" (their own dependency name), the rest in "<name>"
            assert rr and epi == "gelu_pan" and dst.get("q0", 0) == 0 and 3 * split == 24, name
            op["split"] = split
        if h1first:  # round 20 IL: the A fill sends chunks 12-23 first (design8 loads half 1, then half 0)
            assert K == 1536 and A_HALF and len(op["a"][0]) == 1, name
            for fl in op["a"]:
                t = fl[0]
                assert t["sizes"] == [1, 24, 1, t["sizes"][3]], t
                h1 = dict(t, off=t["off"] + 12 * t["strides"][1], sizes=[1, 12, 1, t["sizes"][3]])
                fl[:] = [h1, dict(t, sizes=[1, 12, 1, t["sizes"][3]])]
            op["h1first"] = True
        if reuse_a and A_HALF:
            prev = next(o for o in reversed(self.ops) if o["kind"] in ("gemm", "attn"))
            assert prev["kind"] == "gemm" and prev["src"] is src and prev["K"] == K == 768, (name, "reuse_a needs the previous slot op's A")
            op["reuse_a"] = True
            if prev.get("nres"):
                op["nres"] = True  # (the slot holds the NRES layout: this op's sends read it too)
        if nres:  # round 20 NRES: the A arrives from the norm on chip (no fill, no load); the previous op must be that norm
            assert K == 768 and A_HALF and self.ops[-1]["kind"] == "ew" and self.ops[-1].get("nres") and self.ops[-1]["dst"] is src, name
            op["a"] = [[] for _ in range(COLS)]
            op["nres"] = True
        self.ops.append(op)
        for k in self._keys(dst, K=None, nq=nobj if epi in ("gated", "gelu_pan") else None):
            self.writer[k] = name + "#0" if split and k[1] == 0 else name
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
        if A_HALF:  # the Q objects once, into both halves (pass 0 reads half 0, pass 1 half 1)
            a = [[dict(self.tap("act", qkv["base"] + c * QKV_PER_HC, [1, 12, 3, 3456], [0, QKV_PER_H, 3 * OBJ, 1]), pkt="b")] for c in range(COLS)]
        if A_HALF and P60G:  # dealt column c: slot positions 0-5 (= [tile][half]) from the natural half tiles (see P60G above)
            a = []
            for c in range(COLS):
                if c in (2, 5):
                    t = self.tap("act", qkv["base"] + 2 * QKV_PER_HC + (c == 5) * HQ, [12, 2, 3, HQ], [QKV_PER_H, 3 * QKV_PER_HC, 3 * OBJ, 1])
                else:
                    T, e = divmod(c, 3)
                    t = self.tap("act", qkv["base"] + 3 * T * OBJ + e * HQ, [12, 3, 2, HQ], [QKV_PER_H, 3 * QKV_PER_HC, QKV_PER_HC, 1])
                a.append([dict(t, pkt="b")])
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

    def norm(self, name, mode, srcs, params_w, dst=None, params_bytes=None, nres=False):
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
            params_bytes = kb_pad(3 * SLOT if mode in ("rms", "none") else 5 * SLOT)  # header slot + vectors (AL1: padded)
        fills = [[self.tap("w", params_w, [1, 1, 1, params_bytes], [0, 0, 0, 1]),
                  self.tap("act", srcs[0]["base"] + c * srcs[0]["col"], [RB, 12, max(n_in, 1), XB], [UNIT, KSTEP16, d, 1])] for c in range(COLS)]
        drains = [self.tap("act", dst["base"] + c * dst["col"], [RB, 12, 1, XB if nres else UNIT], [UNIT, KSTEP16, 0, 1]) for c in range(COLS)]
        self.ops.append(dict(kind="ew", fn="norm", name=name, mode=mode, n_in=n_in, params_w=params_w, params_bytes=params_bytes,
                             fills=fills, c=drains, dst=dst, srcs=srcs))
        if nres:  # round 20 NRES: X' only to DDR (1,024 B per unit); the panels go to the memtile A slot
            self.ops[-1]["nres"] = True
        return dst

    def ssm(self, name, area, params_w, params_rot=None):
        """The recurrent block's elementwise part on the row-3 cores, head (k-step) major: per head one params fill
        (SSM_PARAM_BYTES) and one chunk fill of the 7 input streams, one drain for the 6 output streams (round 17 XROT: 3
        streams, h and z, in the rotating phases, with the params copy whose header says so)."""
        base, col = area["base"], area["col"]
        rot = area["rot"] is not None
        pw = params_rot + area["rot"] * 12 * kb_pad(SSM_PARAM_BYTES) if rot else params_w
        ps = kb_pad(SSM_PARAM_BYTES)  # round 17 AL1: the per-head params stride / fill
        fills = [[(self.tap("w", pw + q * ps, [1, 1, 1, ps], [0, 0, 0, 1]),
                   self.tap("act", base + c * col + q * KSTEP16, [RB, 7, 1, XB], [UNIT, S, 0, 1])) for q in range(12)] for c in range(COLS)]
        drains = [self.tap("act", area["out"] + c * col, [12, RB, 3 if rot else 6, XB], [KSTEP16, UNIT, S, 1]) for c in range(COLS)]
        self.ops.append(dict(kind="ew", fn="ssm", name=name, fills=fills, c=drains, dst=area["z"], area=area, params_w=params_w,
                             params_rot=params_rot, pstride=ps))
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
        keep = ("kind", "fn", "name", "K", "nt", "N", "epi", "flags", "nsend", "w", "a", "b", "c", "fills", "passes", "mode", "n_in", "deps", "reuse_a", "rr", "split", "h1first", "nres")
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


MAX_DEPTH = 12


def full_step(depth=12, skip=(), alloc_depth=MAX_DEPTH, phase=None):
    """The whole frame: `depth` blocks and the encoder norm. The host writes X0 (the X' parts) every frame: the
    patch embedding + pos-emb for the image rows (torch's bf16 conv on the CPU, 0.2 ms) and the query tokens.
    Memory (ACT, W) is laid out for `alloc_depth` blocks whatever `depth` is, and the ops of blocks >= depth are
    dropped: every depth shares one layout, one packed W and one xclbin (the final norm reads the same regions after
    any block), so a depth is just another runtime sequence (.bin).
    Returns (plan, info) where info names the regions the host touches (X0, SSM state areas, the final LN rows)."""
    alloc_depth = max(depth, alloc_depth)
    p = Plan()
    r = block_regions(p)
    x0 = p.q16("X0")
    phase = XPH if phase is None else phase
    areas = [p.ssm_area(f"SSM{b}", phase) for b in range(alloc_depth)]
    fin = p.q16("FINAL")
    w = {}
    prev = [x0]  # inputs of block 0's norm0
    for b in range(alloc_depth):
        wb = w[b] = {}
        wb["norm0"] = p.alloc_w(kb_pad(3 * SLOT))
        p.norm(f"b{b}_norm0", "rms", prev, wb["norm0"], dst=r["R0"], nres="norm0" in NRES)
        p.gemm(f"b{b}_yx", r["R0"], 768, 8, epi="gelu", flags=["gelu"] * 12 + ["plain"] * 12, dst=areas[b]["yx"], rr=True, nres="norm0" in NRES)
        wb["ssm"] = p.alloc_w(12 * kb_pad(SSM_PARAM_BYTES))
        if XROT:  # round 17: the four phases' params (header word 2 = 1 + phase)
            wb["ssm_rot"] = p.alloc_w(4 * 12 * kb_pad(SSM_PARAM_BYTES))
        if "ssm" not in skip:
            p.ssm(f"b{b}_ssm", areas[b], wb["ssm"], wb.get("ssm_rot"))
        p.gemm(f"b{b}_out", areas[b]["z"], 768, 4, dst=r["R1"])
        wb["norm1"] = p.alloc_w(kb_pad(3 * SLOT))
        p.norm(f"b{b}_norm1", "rms", [r["R1"], r["R0"]], wb["norm1"], dst=r["R2"], nres="norm1" in NRES)
        p.gemm(f"b{b}_up_a", r["R2"], 768, 16, epi="gated", dst=dict(r["G"], q0=0), nres="norm1" in NRES)
        p.gemm(f"b{b}_up_b", r["R2"], 768, 16, epi="gated", dst=dict(r["G"], q0=24), reuse_a=True)
        p.gemm(f"b{b}_down_a", dict(r["G"], q0=0), 1536, 4, dst=r["D1"], h1first=DAPRE)
        p.gemm(f"b{b}_down_b", dict(r["G"], q0=24), 1536, 4, dst=r["D2"], bias_in=False)
        wb["norm2"] = p.alloc_w(kb_pad(5 * SLOT))
        p.norm(f"b{b}_norm2", "ln", [r["D1"], r["D2"], r["R2"]], wb["norm2"], dst=r["R3"], nres="norm2" in NRES)
        p.gemm(f"b{b}_qkv", r["R3"], 768, 12, epi="qkv", dst=r["QKV"], rr=True, nres="norm2" in NRES)
        if "attn" not in skip:
            p.attn(f"b{b}_attn", r["QKV"], dst=r["O"])
        p.gemm(f"b{b}_out_proj", r["O"], 768, 4, dst=r["R4"], deal=P60G)
        wb["norm3"] = p.alloc_w(kb_pad(5 * SLOT))
        p.norm(f"b{b}_norm3", "ln", [r["R4"], r["R3"]], wb["norm3"], dst=r["R5"], nres="norm3" in NRES)
        p.gemm(f"b{b}_mlp0", r["R5"], 768, 16, epi="gelu_pan", dst=dict(r["M"], q0=0), rr=IL, split=8 if IL else 0, nres="norm3" in NRES)
        p.gemm(f"b{b}_mlp3_a", dict(r["M"], q0=0), 1536, 4, dst=r["D3"], h1first=IL)
        p.gemm(f"b{b}_mlp3_b", dict(r["M"], q0=24), 1536, 4, dst=r["D4"], bias_in=False)
        prev = [r["D3"], r["D4"], r["R5"]]
    w["final"] = p.alloc_w(kb_pad(5 * SLOT))
    p.norm("final", "ln_out", prev, w["final"], dst=fin)
    p.ops = [op for op in p.ops if op["name"] == "final" or int(op["name"].split("_")[0][1:]) < depth]
    info = dict(x0=x0, areas=areas[:depth], final=fin, w=w, regions=r)
    return p, info


if __name__ == "__main__":
    p, info = full_step()
    print(f"ACT {p.act_bytes / 1e6:.1f} MB, W {p.w_bytes / 1e6:.1f} MB, ops {len(p.ops)}")
    print(f"sched json {len(p.sched_json()) / 1e3:.0f} KB")
