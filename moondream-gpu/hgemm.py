"""Hand-written HIP WMMA GEMMs (hip/i8gemm.hip -> hip/libmdgemm.so) exposed as drop-in replacements for the Triton
linears in tgemm.py: HI8Lin (int8 W8A8, per-token/per-channel scales, fused bias+gelu) and HLin (bf16, fused bias+gelu).
The library is built in the ROCm 7.14 container (see README.md); it only depends on libamdhip64.so.7, which torch's
rocm wheel already loads, so it works in the host venv-rocm714 too."""
import ctypes, os, time, statistics, torch
import tgemm
from tgemm import quant_rows as _quant_rows_contig
import triton

def quant_rows(x2d, pad=None):
    """per-token int8 quantisation into a row-padded buffer (stride K + pad bytes) so that activation rows do not alias DRAM channels either"""
    pad = WPAD if pad is None else pad
    M, K = x2d.shape
    qb = torch.empty(M, K + pad, device=x2d.device, dtype=torch.int8); s = torch.empty(M, device=x2d.device, dtype=torch.float32)
    tgemm._quant_rows[(M,)](x2d, qb, s, K, x2d.stride(0), K + pad, BN=triton.next_power_of_2(K), num_warps=8)
    return qb[:, :K], s

_DIR = os.path.dirname(os.path.abspath(__file__))
_lib = None

def lib():
    global _lib
    if _lib is None:
        torch.cuda.init()
        _lib = ctypes.CDLL(os.path.join(_DIR, "hip", "libmdgemm.so"))
        P = ctypes.c_void_p; I = ctypes.c_int
        _lib.i8gemm_launch.argtypes = [I, P, P, P, P, P, P, P, I, I, I, I, I, I, I, I]; _lib.i8gemm_launch.restype = I
        _lib.bf16gemm_launch.argtypes = [I, P, P, P, P, P, I, I, I, I, I, I, I, I]; _lib.bf16gemm_launch.restype = I
        _lib.i8gemm_ncfg.restype = I; _lib.bf16gemm_ncfg.restype = I
        if hasattr(_lib, "i8gemm_small_launch"):
            _lib.i8gemm_small_launch.argtypes = [I, P, P, P, P, P, P, P, P, I, I, I, I, I, I, I]; _lib.i8gemm_small_launch.restype = I
            _lib.i8gemm_small_ncfg.restype = I
    return _lib

def _stream():
    return torch.cuda.current_stream().cuda_stream

_cfg_cache = {}
SWAP = 1

def _time(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return min(ts)

# fixed choices from the C++ microbenchmarks (hip/i8test, hip/bf16test; see RESULTS.md); anything else is autotuned once
FIXED = {"i8": lambda M, N, K: 27,                                   # 128x128 tile, BK=128, 8 waves of 32x64, single LDS buffer
         "i8s": lambda M, N, K: 3 if PACK else (4 if (N <= 2048 and K <= 2048) else 6),  # split-K: 3 = BN=128 BK=128 (packed), 4/6 = BN=64 BK=128/256
         "bf16": lambda M, N, K: 1 if K <= 1152 else 0}
def pick_cfg(kind, M, N, K, run, ncfg):
    """config for this (kind, M, N, K): fixed table above, else a tiny autotune cached in memory and in hip/cfgcache.txt"""
    key = (kind, M, N, K)
    if key in _cfg_cache: return _cfg_cache[key]
    if kind in FIXED and not os.environ.get("HGEMM_AUTOTUNE"):
        c = FIXED[kind](M, N, K)
        if run(c) == 0:
            _cfg_cache[key] = c; return c
    path = os.path.join(_DIR, "hip", "cfgcache.txt")
    if os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) == 5 and (p[0], int(p[1]), int(p[2]), int(p[3])) == key:
                _cfg_cache[key] = int(p[4]); return _cfg_cache[key]
    best, bt = None, 1e9
    for c in range(ncfg):
        if run(c) != 0: continue
        t = _time(lambda: run(c))
        if t < bt: best, bt = c, t
    _cfg_cache[key] = best
    try:
        with open(path, "a") as f: f.write(f"{kind} {M} {N} {K} {best}\n")
    except OSError:
        pass
    return best

_ws = None
def workspace():
    """int32 split-K partials for the small-M kernel (shared by all layers: kernels on one stream serialise)"""
    global _ws
    if _ws is None: _ws = torch.empty(16 * 64 * 8192, device="cuda", dtype=torch.int32)
    return _ws
SMALL_M = 64  # M <= SMALL_M uses the weight-streaming split-K kernel
WPAD = int(os.environ.get("HGEMM_WPAD", "64"))  # bytes of padding per int8 weight row (unpacked layout)
PACK = os.environ.get("HGEMM_PACK", "1") == "1"   # tiled weight layout: each (TN rows x TK) block contiguous -> one 16 KB DRAM-friendly chunk per K step
TN, TK_I8, TK_BF16 = 128, 128, 64

def pack_tiles(w, tn, tk):
    """(N,K) -> [ceil(N/tn)][ceil(K/tk)][tn][tk] contiguous, zero padded"""
    N, K = w.shape; Np = (N + tn - 1) // tn * tn; Kp = (K + tk - 1) // tk * tk
    wp = torch.zeros(Np, Kp, device=w.device, dtype=w.dtype); wp[:N, :K] = w
    return wp.view(Np // tn, tn, Kp // tk, tk).permute(0, 2, 1, 3).contiguous()

class HI8Lin:
    """W8A8 int8 (per-channel weight scale, per-token dynamic activation scale), weights kept in the natural (N,K) layout."""
    fused_gelu = True
    fused_res = False
    def __init__(self, weight, bias, dtype):
        w = weight.detach().float()
        self.N, self.K = w.shape
        s = w.abs().amax(dim=1).clamp_min(1e-8) / 127.0
        wq = (w / s[:, None]).round().clamp(-127, 127).to(torch.int8)
        # row stride padded by WPAD bytes: with K a power of two, rows at the same k offset would all map to the same DRAM channel
        self.packed = PACK and self.N % TN == 0 and self.K % TK_I8 == 0
        if self.packed:
            self.wq = pack_tiles(wq, TN, TK_I8); self.ldw = self.K
        else:
            self.ldw = self.K + WPAD
            self.wq = torch.zeros(self.N, self.ldw, device=w.device, dtype=torch.int8); self.wq[:, :self.K] = wq  # (N, ldw)
        self.sw = s.contiguous()
        self.b = None if bias is None else bias.detach().float().contiguous()
        self.dtype = dtype
        self.cfgs = {}

    def quant(self, x):
        return quant_rows(x.reshape(-1, self.K))

    def _run(self, cfg, q, sx, c, M, gelu):
        if M <= SMALL_M and hasattr(lib(), "i8gemm_small_launch"):
            return lib().i8gemm_small_launch(cfg, _stream(), q.data_ptr(), self.wq.data_ptr(), workspace().data_ptr(), sx.data_ptr(), self.sw.data_ptr(),
                                             self.b.data_ptr() if self.b is not None else None, c.data_ptr(), M, self.N, self.K, int(gelu), q.stride(0), self.ldw, int(self.packed))
        return lib().i8gemm_launch(cfg, _stream(), q.data_ptr(), self.wq.data_ptr(), sx.data_ptr(), self.sw.data_ptr(),
                                   self.b.data_ptr() if self.b is not None else None, c.data_ptr(), M, self.N, self.K, int(gelu), SWAP, q.stride(0), self.ldw, int(self.packed))

    def __call__(self, x, gelu=False, res=None, pre=None):
        shp = x.shape; x2 = x.reshape(-1, self.K); M = x2.shape[0]
        q, sx = pre if pre is not None else quant_rows(x2)
        c = torch.empty(M, self.N, device=x.device, dtype=x.dtype)
        cfg = self.cfgs.get(M)
        if cfg is None:
            cfg = self.cfgs[M] = pick_cfg("i8s" if M <= SMALL_M else "i8", M, self.N, self.K, lambda cc: self._run(cc, q, sx, c, M, gelu),
                                          lib().i8gemm_small_ncfg() if M <= SMALL_M else lib().i8gemm_ncfg())
        rc = self._run(cfg, q, sx, c, M, gelu)
        assert rc == 0, rc
        out = c.reshape(*shp[:-1], self.N)
        return out if res is None else out + res

class HLin:
    """bf16 GEMM with fused bias (+gelu); weights in (N,K) layout."""
    fused_gelu = True
    fused_res = False
    def __init__(self, weight, bias, dtype):
        assert dtype == torch.bfloat16
        w = weight.detach().to(dtype)
        self.N, self.K = w.shape
        self.packed = PACK and self.K % 8 == 0
        if self.packed:
            self.w = pack_tiles(w, TN, TK_BF16); self.ldw = self.K
        else:
            self.ldw = self.K + WPAD // 2  # padded row stride (elements)
            self.w = torch.zeros(self.N, self.ldw, device=w.device, dtype=dtype); self.w[:, :self.K] = w
        self.b = None if bias is None else bias.detach().float().contiguous()
        self.cfgs = {}
        self.ok = self.K % 8 == 0
        if not self.ok:  # e.g. patch embedding K=588
            self.wt = w.contiguous()
            self.bb = None if bias is None else bias.detach().to(dtype).contiguous()

    def _run(self, cfg, x2, c, M, gelu):
        return lib().bf16gemm_launch(cfg, _stream(), x2.data_ptr(), self.w.data_ptr(), self.b.data_ptr() if self.b is not None else None,
                                     c.data_ptr(), M, self.N, self.K, int(gelu), SWAP, x2.stride(0), self.ldw, int(self.packed))

    def __call__(self, x, gelu=False, res=None):
        shp = x.shape; x2 = x.reshape(-1, self.K).contiguous(); M = x2.shape[0]
        if not self.ok:
            out = torch.nn.functional.linear(x2, self.wt, self.bb)
            if gelu: out = torch.nn.functional.gelu(out, approximate="tanh")
            out = out.reshape(*shp[:-1], self.N)
            return out if res is None else out + res
        c = torch.empty(M, self.N, device=x.device, dtype=x.dtype)
        cfg = self.cfgs.get(M)
        if cfg is None:
            cfg = self.cfgs[M] = pick_cfg("bf16", M, self.N, self.K, lambda cc: self._run(cc, x2, c, M, gelu), lib().bf16gemm_ncfg())
        rc = self._run(cfg, x2, c, M, gelu)
        assert rc == 0, rc
        out = c.reshape(*shp[:-1], self.N)
        return out if res is None else out + res
