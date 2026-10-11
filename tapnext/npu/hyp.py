import numpy as np
d = np.load("/work/npu-tapnext/build/t4_dbg.npz")
got, q, k, vT, o_ref = d["got"], d["q"], d["k"], d["vT"], d["o_ref"]
M=1072; g=got[:M]
def corr(a,b): return np.corrcoef(a.ravel(), b.ravel())[0,1]
def attn(qh,kh,vh,keys,scale=0.125):
    s=qh@kh[keys].T*scale; s=s-s.max(1,keepdims=True); p=np.exp(s); return (p@vh[keys])/p.sum(1,keepdims=True)
def allheads(keys, **kw):
    cs=[]
    for h in range(12):
        sl=slice(h*64,h*64+64); cs.append(corr(g[:,sl], attn(q[:,sl],k[:,sl],vT[:,sl],keys,**kw)[:M]))
    return np.mean(cs)
print("saved o_ref corr", corr(g, o_ref[:M]))
print("48 keys", allheads(np.arange(48)), "1072 keys", allheads(np.arange(1072)))
for n in (8,16,24,32,40,48):
    print("first", n, "keys:", round(allheads(np.arange(n)),3))
for b in range(6):
    keys=np.array([i for i in range(48) if i//8!=b]); print("drop block", b, round(allheads(keys),3))
for pr in range(3):
    print("only block pair", pr, round(allheads(np.arange(16*pr,16*pr+16)),3))
for sc in (0.125*0.5, 0.125*0.75, 0.125, 0.125*1.25):
    print("scale", sc, round(allheads(np.arange(48), scale=sc),3))
rows=np.arange(M)
for i in range(8):
    sel=rows%8==i
    cs=[corr(g[sel,h*64:(h+1)*64], o_ref[:M][sel,h*64:(h+1)*64]) for h in range(12)]
    print("row%8 =",i,"corr",round(np.mean(cs),3))
# per column / tile
for c in range(8):
    sel=rows//144==c
    print("col",c,"corr",round(corr(g[sel], o_ref[:M][sel]),3))
