"""Fit the recurrence-gate direction at the QUERY ARROW from the length-matched slotgate twins (D66a dumps).
g_L = mean over pairs of (x_two - x_one) at layer L (query arrow = last dumped position); report held-out separation (leave-one-pair-out sign test),
natural gap (mean projection difference), cosines to the class direction u (sentiment, if available) and save dirs_D66_gate_4B.npz (unit vectors + gap per layer)."""
import numpy as np,glob,json,sys,os
d=sys.argv[1]; OUT=sys.argv[2] if len(sys.argv)>2 else 'dirs_D66_gate_4B.npz'; files=sorted(glob.glob(f'{d}/resid_*.npz'),key=lambda p:int(p.split('_')[-1][:-4]))
rows={}
for p in files:
    z=np.load(p); k=str(z['kind']); t=str(z['text']); rows.setdefault(t.split('\n')[-1],{})[k]=z   # pair by the query line
pairs=[v for v in rows.values() if len(v)==2]; print('pairs',len(pairs))
Ls=[int(k[1:]) for k in pairs[0]['icl_slotgate_one'].files if k.startswith('l')]
out={}
for L in Ls:
    one=np.stack([v['icl_slotgate_one'][f'l{L}'][-1] for v in pairs]); two=np.stack([v['icl_slotgate_two'][f'l{L}'][-1] for v in pairs])
    D=two-one; g=D.mean(0); g/=np.linalg.norm(g)
    # leave-one-out: sign of (D_i . g_-i)
    ho=[]
    for i in range(len(D)):
        gi=np.delete(D,i,0).mean(0); gi/=np.linalg.norm(gi); ho.append(float(D[i]@gi>0))
    gap=float((D@g).mean()); nrm=float(np.linalg.norm(one,axis=1).mean())
    cos=float(np.mean([np.dot(D[i]/np.linalg.norm(D[i]),D[j]/np.linalg.norm(D[j])) for i in range(len(D)) for j in range(i+1,len(D))]))
    out[L]=(g,gap); print(f"L{L:2d} held-out sign {np.mean(ho):.2f}  gap {gap:6.2f}  |x| {nrm:6.1f}  gap/|x| {gap/nrm:.3f}  pairwise cos of per-item deltas {cos:.2f}")
np.savez(OUT,**{f'l{L}':out[L][0] for L in Ls},**{f'gap{L}':np.array(out[L][1]) for L in Ls}); print('saved',OUT)
