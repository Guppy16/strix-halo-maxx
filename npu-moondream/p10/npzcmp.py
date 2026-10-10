import sys, numpy as np
a = np.load(sys.argv[1])["o"]; b = np.load(sys.argv[2])["o"]
print("outputs identical:", bool(np.array_equal(a, b)), "max |diff|", float(np.abs(a - b).max()), "shape", a.shape)
