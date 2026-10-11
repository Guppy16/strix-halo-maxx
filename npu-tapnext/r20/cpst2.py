import sys
import numpy as np
for fn in sys.argv[1:]:
    d = np.load(fn); a, n = d["a"], d["names"]
    m = a[n == "b11_mlp3_a"]; f = m[m[:, 10] == m[:, 10].min()]
    p = a[n == "b11_mlp0"]; L = p[p[:, 10] == p[:, 10].max()]; kend = (L[:, 9] + L[:, 3] + L[:, 4]).max()
    j = np.where(f[:, 12] == 1)[0][0]
    k0 = f[:, 9] + f[:, 3]
    print(f"{fn}: flag seen {(f[j,11]-kend)/1810:6.2f} us after mlp0's last k-step end; core 0_2 k0 end {(k0[j]-kend)/1810:6.2f}; median k0 {(np.median(k0)-kend)/1810:6.2f}")
