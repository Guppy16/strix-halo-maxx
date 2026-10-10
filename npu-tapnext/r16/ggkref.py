"""round 16: host reference for r16/ggkb.cc's first output block (gate_x, np 0, column block 0): the same LCG inputs,
exact products, and which of the device results (ggk / gate_gemm) matches which hypothesis"""
import numpy as np


class R:
    def __init__(self, seed):
        self.s = (11 + seed) & 0xFFFFFFFF

    def __call__(self):
        self.s = (self.s * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.s >> 8


def inputs(seed, e0):
    r = R(seed)
    w = np.array([(e0 + r() % 6) if i % 9 == 0 else (r() & 0xFF) for i in range(9216)], np.uint8)
    a = np.array([(e0 + r() % 6) if i % 9 == 0 else (r() & 0xFF) for i in range(576)], np.uint8)
    tb = []
    for t in range(2):
        tb.append(np.array([((r() % 2001) - 1000) / 256 for _ in range(8 * 64)], np.float32))
    return w, a, tb


def panel(b):  # 72 B -> [8 rows][8] values
    g = b.reshape(8, 9)
    return g[:, 1:].view(np.int8).astype(np.float64) * np.exp2(g[:, :1].astype(np.float64) - 133)


def bf16(x):
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def block(w, a, tb, np_=0, c=0, variant="nt"):
    acc = np.zeros((8, 8))
    for kk in range(8):
        A = panel(a[72 * kk:72 * kk + 72])
        B = panel(w[np_ * 1152 + (2 * kk + c) * 72:][:72])
        acc += A @ B.T if variant == "nt" else A @ B
    r = bf16(acc.astype(np.float32)).astype(np.uint32) << 16
    s = r.view(np.float32).astype(np.float64).reshape(-1) + tb[0][np_ * 128 + c * 64:][:64]
    return bf16(s.astype(np.float32))


def hyp(w, a, tb, got):
    """count matches of ggk's first block against structural variants"""
    def run(fa, fb, kks, bias=True):
        acc = np.zeros((8, 8))
        for kk in kks:
            acc += panel(a[72 * fa(kk):][:72]) @ panel(w[fb(kk):][:72]).T
        r = (bf16(acc.astype(np.float32)).astype(np.uint32) << 16).view(np.float32).astype(np.float64).reshape(-1)
        return bf16((r + (tb[0][:64] if bias else 0)).astype(np.float32))
    H = {"exact": run(lambda k: k, lambda k: 144 * k, range(8)), "nobias": run(lambda k: k, lambda k: 144 * k, range(8), False),
         "Bswap": run(lambda k: k, lambda k: 144 * k + 72, range(8))}
    for j in range(8):
        H[f"miss{j}"] = run(lambda k: k, lambda k: 144 * k, [k for k in range(8) if k != j])
    for s in (1, 7):
        H[f"Ashift{s}"] = run(lambda k, s=s: (k + s) % 8, lambda k: 144 * k, range(8))
    for s in (72, -72, 144):
        H[f"Bshift{s}"] = run(lambda k: k, lambda k, s=s: max(0, 144 * k + s), range(8))
    H["Bseq"] = run(lambda k: k, lambda k: 72 * k, range(8))
    return {k: int((v == got).sum()) for k, v in H.items()}


def hyp2(w, a, tb, got):
    """M50 (chain 0, kk 5) reading ex2 one cycle before A6's pop lands in it: whole panel, or the exponent / mantissa part only"""
    def mix(pm, pe):  # mantissas of pm, exponents of pe (72 B panels)
        out = pm.copy().reshape(8, 9)
        out[:, 0] = pe.reshape(8, 9)[:, 0]
        return out.reshape(-1)

    def run(b5):
        acc = np.zeros((8, 8))
        for kk in range(8):
            B = b5 if kk == 5 else w[144 * kk:][:72]
            acc += panel(a[72 * kk:][:72]) @ panel(B).T
        r = (bf16(acc.astype(np.float32)).astype(np.uint32) << 16).view(np.float32).astype(np.float64).reshape(-1)
        return bf16((r + tb[0][:64]).astype(np.float32))
    b50, a6 = w[144 * 5:][:72], a[72 * 6:][:72]
    H = {"B50=A6": run(a6), "mant B50, exp A6": run(mix(b50, a6)), "mant A6, exp B50": run(mix(a6, b50))}
    return {k: int((v == got).sum()) for k, v in H.items()}
