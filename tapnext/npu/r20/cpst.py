"""Round 20: the CPST probe (r20c2): core 0_2's stamp when the CP passed b11_mlp3_a's dependency wait vs the TPROF times.
usage: python3 r20/cpst.py X.npz"""
import sys
import numpy as np
d = np.load(sys.argv[1]); a, names = d["a"], d["names"]
G = 1810.0
def recs(nm):
    return a[names == nm]
mlp0, m3 = recs("b11_mlp0"), recs("b11_mlp3_a")
n0 = m3[:, 10].min()
f = m3[m3[:, 10] == n0]
tcp = f[:, 11]
tcp = tcp[tcp != 0]
print("records with a CP stamp:", len(tcp), np.unique(tcp)[:4])
tc = np.median(tcp)
# mlp0 tiles: t0 + first + ks = mm_ktile end (mid / epi not recorded in this build); last tile start / end
last_n = mlp0[:, 10].max()
L = mlp0[mlp0[:, 10] == last_n]
kend = L[:, 9] + L[:, 3] + L[:, 4]
print(f"mlp0 last tile: k-steps end (max over cores) {kend.max()}  spread {kend.max()-kend.min()} cyc")
k0 = f[:, 9] + f[:, 3]
ksk = np.median(f[:, 4] / (f[:, 5] - 1))
print(f"mlp3_a tile 0: k0 end median {np.median(k0):.0f}, ks/k {ksk:.0f}; tile entry t0 median {np.median(f[:,9]):.0f}")
print(f"  CP passed the wait at {tc:.0f}: {(tc - kend.max())/G:.2f} us after mlp0's last k-step end (+ epilogue ~4,910 cyc = {4910/G:.2f} us)")
print(f"  k0 end - CP pass: median {(np.median(k0) - tc)/G:.2f} us, min {(k0.min()-tc)/G:.2f}, max {(k0.max()-tc)/G:.2f}")
