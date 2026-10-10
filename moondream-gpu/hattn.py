"""ctypes wrapper for the HIP WMMA flash-attention prefill kernel (hip/attn.hip -> hip/libmdattn.so), mask-free, non-causal."""
import ctypes, os, torch
_DIR = os.path.dirname(os.path.abspath(__file__))
_lib = None
def lib():
    global _lib
    if _lib is None:
        torch.cuda.init()
        _lib = ctypes.CDLL(os.path.join(_DIR, "hip", "libmdattn.so"))
        P = ctypes.c_void_p; I = ctypes.c_int; L = ctypes.c_long
        _lib.attn_fwd.argtypes = [P, P, P, P, P, I, I, I, I, I, L, L, L, L, L, ctypes.c_float]; _lib.attn_fwd.restype = I
        _lib.attn_fwd2.argtypes = _lib.attn_fwd.argtypes; _lib.attn_fwd2.restype = I
        _lib.attn_fwd3.argtypes = _lib.attn_fwd.argtypes; _lib.attn_fwd3.restype = I
        _lib.vtranspose.argtypes = [P, P, P, I, I, I, I, I, L, L]; _lib.vtranspose.restype = I
    return _lib

class Attn:
    """holds the padded V^T buffer for one (H, D, T) problem; call(q, k, v) -> (T, H*D) bf16"""
    def __init__(self, H, T, D, device="cuda"):
        self.H, self.T, self.D = H, T, D
        self.DP = 64 if D <= 64 else 80
        self.TP = (T + 63) // 64 * 64
        self.vt = torch.zeros(H, self.DP, self.TP, device=device, dtype=torch.bfloat16)
        self.scale = D ** -0.5
        self.version = int(os.environ.get('HATTN_V', '2'))
    def __call__(self, q, k, v, out=None):
        """q, k: (1, H, T, D) views (any strides along H/T, contiguous D); v: (1, H, T, D) view. returns (1, T, H*D)"""
        H, T, D = self.H, self.T, self.D
        assert q.shape == (1, H, T, D) and q.stride(3) == 1 and k.stride(3) == 1
        assert v.stride(3) == 1
        lib().vtranspose(torch.cuda.current_stream().cuda_stream, v.data_ptr(), self.vt.data_ptr(), T, self.TP, H, D, self.DP, v.stride(1), v.stride(2))
        if out is None: out = torch.empty(1, T, H * D, device=q.device, dtype=torch.bfloat16)
        fn = {1: lib().attn_fwd, 2: lib().attn_fwd2, 3: lib().attn_fwd3}[self.version]
        rc = fn(torch.cuda.current_stream().cuda_stream, q.data_ptr(), k.data_ptr(), self.vt.data_ptr(), out.data_ptr(), T, self.TP, H, D, self.DP,
                            q.stride(1), q.stride(2), k.stride(1), k.stride(2), out.stride(1), self.scale)
        assert rc == 0, rc
        return out

if __name__ == "__main__":
    import time, statistics, torch.nn.functional as F
    def bench(fn, n=30):
        for _ in range(3): fn()
        torch.cuda.synchronize(); ts = []
        for _ in range(n):
            t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
        return statistics.median(ts) * 1e6
    torch.manual_seed(0)
    for H, T, D, strided in ((32, 730, 64, False), (16, 729, 72, True)):
        if strided:  # vision layout: qkv (1, T, 3*H*D) -> chunk -> view (1, T, H, D) -> transpose(1, 2)
            qkv = torch.randn(1, T, 3 * H * D, device="cuda", dtype=torch.bfloat16)
            q, k, v = [t.view(1, T, H, D).transpose(1, 2) for t in qkv.chunk(3, dim=-1)]
        else:
            q, k, v = [torch.randn(1, H, T, D, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        ref = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, T, H * D)
        ref32 = F.scaled_dot_product_attention(q.float(), k.float(), v.float()).transpose(1, 2).reshape(1, T, H * D)
        usd = bench(lambda: F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, T, H * D))
        for ver in (2, 3):
            a = Attn(H, T, D); a.version = ver
            out = a(q, k, v)
            d = (out.float() - ref.float()); d32 = (out.float() - ref32)
            vt_copy = bench(lambda: lib().vtranspose(torch.cuda.current_stream().cuda_stream, v.data_ptr(), a.vt.data_ptr(), T, a.TP, H, D, a.DP, v.stride(1), v.stride(2)))
            assert torch.equal(a.vt[:, :D, :T], v[0].transpose(1, 2)), "transpose mismatch"
            print(f"v{ver} H={H} T={T} D={D}: maxabs vs sdpa {d.abs().max().item():.4f} rel {d.norm().item()/ref.float().norm().item():.5f} | vs fp32: HIP rel {d32.norm().item()/ref32.norm().item():.5f}, sdpa rel {(ref.float()-ref32).norm().item()/ref32.norm().item():.5f} | "
                  f"sdpa {usd:.1f} us | HIP {bench(lambda: a(q, k, v)):.1f} us incl. V^T copy ({vt_copy:.1f} us)", flush=True)
