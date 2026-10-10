import sys
import numpy as np
from aie.utils import bfp
from ml_dtypes import bfloat16
exec(open("/work/p9/model4.py").read().split("HH = 4")[0].split("xs = np.linspace")[0])  # helpers + data
HH = 2
for h in range(HH):
    Qb, Kb, Vb = rq(Q[h]), rq(K[h]), rq(Vp[h].T).T[:NK]
    s = bf(bf(Qb @ Kb.T) * scale)
    e = exp_srs_np(bf(s - s.max(axis=1, keepdims=True)))
    P = rq(e, "floor"); r = bf(1.0 / P.sum(axis=1, keepdims=True, dtype=np.float32))
    got = bf((P @ Vb) * r)
    alpha = (o[h] * got).sum(axis=1) / (got * got).sum(axis=1)   # per-row scale fit
    resid = o[h] - alpha[:, None] * got
    print(f"head {h}: per-row scale alpha mean {alpha.mean():.5f} std {alpha.std():.2e} min {alpha.min():.4f} max {alpha.max():.4f}; "
          f"rms_rel total {np.sqrt(((o[h]-got)**2).sum()/(got**2).sum()):.2e} after per-row scale {np.sqrt((resid**2).sum()/(got**2).sum()):.2e}")
    # does the scale error correlate with the row's softmax sum or with the number of max updates?
    ssum = e.sum(axis=1); print(f"   corr(alpha, 1/sum) {np.corrcoef(alpha, 1/ssum)[0,1]:+.3f}; corr(alpha, max s) {np.corrcoef(alpha, s.max(axis=1))[0,1]:+.3f}; alpha by query tile: {np.array2string(np.array([alpha[t*48:(t+1)*48].mean() for t in range(16)]), precision=4)}")
    # error vs. the unnormalised numerator? compare o*sum vs P@Vb
    print(f"   rows 0..5 alpha {np.array2string(alpha[:6], precision=4)}; sums {np.array2string(ssum[:6], precision=1)}")
