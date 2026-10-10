# emulate the g_srb / g_maxb overlap: the running-max entry (t*6+rb) aliases scores block 3 + e//4, rows 2(e%4), 2(e%4)+1
import numpy as np
from ml_dtypes import bfloat16
d = np.load("/work/npu-tapnext/build/t4_dbg.npz")
got, q, k, vT, o_ref = d["got"], d["q"], d["k"], d["vT"], d["o_ref"]
M=1072; g=got[:M]
bf=lambda x: np.asarray(x,np.float32).astype(bfloat16).astype(np.float32)
def corr(a,b): return np.corrcoef(a.ravel(), b.ravel())[0,1]
NK=48
def emu(qh,kh,vh,overlap=True):
    out=np.zeros((1152,64))
    for col in range(8):
        for t in range(3):
            for rb in range(6):
                r0=col*144+t*48+rb*8
                s=bf(bf(qh[r0:r0+8]@kh[:NK].T)*0.125)  # [8 rows][48 keys]
                s=s.reshape(8,6,8).transpose(1,0,2).copy()  # [cb][row][key]
                rowmax=s.max(axis=(0,2))  # [8]
                e=(t%2)*6+rb if False else (min(t,1) if t<2 else 0)*6+rb  # pass 0: tiles 0,1 -> t; pass 1: tile 2 -> slot 0
                e=(t if t<2 else 0)*6+rb
                if overlap:
                    cbp=3+e//4; a=2*(e%4)
                    m_old=np.concatenate([s[cbp,a],s[cbp,a+1]])
                    m_new=np.maximum(np.concatenate([rowmax,rowmax]),m_old)
                    if (m_old<m_new).any():
                        s[cbp,a]=m_new[:8]; s[cbp,a+1]=m_new[8:]
                    mb=m_new[:8]
                else:
                    mb=rowmax
                p=np.exp(s-mb[None,:,None])  # [cb][row][key]
                p=p.transpose(1,0,2).reshape(8,NK)
                out[r0:r0+8]=(p@vh[:NK])/p.sum(1,keepdims=True)
    return out[:M]
for ov in (False, True):
    cs=[]
    for h in range(12):
        sl=slice(h*64,h*64+64); cs.append(corr(g[:,sl], emu(q[:,sl],k[:,sl],vT[:,sl],ov)))
    print("overlap" if ov else "clean", "corr", round(np.mean(cs),4), [round(c,3) for c in cs])
