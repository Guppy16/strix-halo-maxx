import numpy as np
from numpy.polynomial import chebyshev as C
def gelu(x): return 0.5*x*(1+np.tanh(0.7978845608*(x+0.044715*x**3)))
def bf(v):  # bf16 RNE
    u = np.asarray(v, np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return r.astype(np.uint32).view(np.float32)
X = np.linspace(-4, 4, 20001)
s = 0.5*(1+np.tanh(0.7978845608*(X+0.044715*X**3)))   # sigmoid(2u)
# s = 0.5 + x*Q(x^2): fit Q(t) on t in [0,16] as a chebyshev series in t
t = X**2; Q = np.where(np.abs(X) > 1e-6, (s-0.5)/np.where(np.abs(X)>1e-6, X, 1), 0.3989)
for deg in range(4, 13):
    # weighted least squares on Q with weight |x| so the error in s=x*Q is uniform
    tt = 2*t/16-1
    V = np.polynomial.chebyshev.chebvander(tt, deg)
    w = np.abs(X)+0.05
    coef, *_ = np.linalg.lstsq(V*w[:,None], Q*w, rcond=None)
    Qf = V @ coef
    sf = 0.5 + X*Qf
    g = X*sf
    # bf16 emulation: horner in power basis with bf16 rounding of p each step, t bf16, x bf16
    pc = C.cheb2poly(coef)  # power basis in tt
    # convert to power basis in t: tt = t/8 - 1
    pt = np.polynomial.polynomial.Polynomial(pc)(np.polynomial.polynomial.Polynomial([-1, 1/8]))
    pt = pt.coef
    xb = bf(X); tb = bf(xb*xb)
    p = np.full_like(xb, pt[-1])
    for c in pt[-2::-1]:
        p = bf(c + tb*p)        # fp32 MAC then bf16 rounding
    sb = bf(0.5 + xb*p)
    gb = bf(xb*sb)
    print(f"deg {deg}: fp32 fit max|g err| {np.abs(g-gelu(X)).max():.2e}  bf16-horner max|g err| {np.abs(gb-gelu(X)).max():.2e} rms {np.sqrt(np.mean((gb-gelu(X))**2)):.2e}  ulp-flips {np.mean(gb != bf(gelu(X))):.3f}")
