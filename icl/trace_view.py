"""Token observatory viewer. usage: python trace_view.py OUT/trace/000_none.npz [OUT/trace/000_circ_only.npz] [--pos -1] [--top 6]
Prints, for one position (default: the query arrow), layer by layer: residual norm, projection on the repeat-flag direction U_l and the gate g,
the heads with the largest write norm there and where each of them attends (top sources); with a second shard, the per-layer differences."""
import sys,numpy as np
fs=[a for a in sys.argv[1:] if a.endswith('.npz')]; pos=int(next((a.split('=')[1] for a in sys.argv if a.startswith('--pos=')),'-1')); top=int(next((a.split('=')[1] for a in sys.argv if a.startswith('--top=')),'6'))
Z=[np.load(f,allow_pickle=True) for f in fs]; z=Z[0]; T=int(z['T']); p=pos if pos>=0 else T+pos; toks=z['tokstr']; late=list(z['late'])
def where(att,l,h,p,k=3):
    row=att[l,h,p,:p+1].astype(float); idx=np.argsort(-row)[:k]; return ' '.join(f"{int(i)}{str(toks[i])!r}:{row[i]:.2f}" for i in idx)
print(f"{fs[0]}  kind={z['kind']} target={z['target']} cond={z['cond']} T={T}  position {p} = {str(toks[p])!r}")
hs=z['hs'].astype(np.float32); L=hs.shape[0]-1; nh=z['att'].shape[1]
for l in range(L):
    r=hs[l+1,p]; n=np.linalg.norm(r); pu=z['proj_u'][late.index(l),p] if l in late else float('nan'); pg=z['proj_g'][l,p] if z['proj_g'].size else float('nan')
    wn=z['wnorm'][l,p].astype(float); hh=np.argsort(-wn)[:top]
    line=f"L{l:2d} |res|={n:7.1f} U={pu:+6.2f} g={pg:+6.2f} | "+'  '.join(f"h{h}:{wn[h]:.1f}[{where(z['att'],l,h,p,2)}]" for h in hh)
    if len(Z)>1:
        d=np.linalg.norm(Z[1]['hs'][l+1,p].astype(np.float32)-r); line+=f"  || Δres vs {Z[1]['cond']}: {d:.1f}"
    print(line)
