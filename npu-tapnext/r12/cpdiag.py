import sys; sys.path.insert(0,'../r11'); sys.path.insert(0,'.')
import gen_gelu as g, cpsched
from msched import Loop
rel=sys.argv[1]; ii=int(sys.argv[2]); bx=int(sys.argv[3]); tl=float(sys.argv[4]) if len(sys.argv)>4 else 60
lp=Loop(g.ops(None,True,rel), consts={"x":[8,9,10,11]}, classes={"x":g.X_FREE,"dm":g.DM_FREE})
print(lp.res_mii(detail=True))
r=cpsched.solve(lp, ii, budgets={"x":bx,"dm":5}, time_limit=tl, workers=4, w_len=int(sys.argv[6]) if len(sys.argv)>6 else 1, H=int(sys.argv[5]) if len(sys.argv)>5 else None)
print(r[1] if r else None)
if r: print(sorted(r[0].items(), key=lambda a:a[1]))
