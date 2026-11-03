"""Fig 19: token observatory. One 12-shot prompt through Qwen3-4B from the D179 trace shards (ICL_DATA/trace/000_{none,circ_only}.npz):
(a) summed attention-head write norm at every (layer, position); (b) at the query arrow, residual projection on the gate direction g and on the
repeat-flag direction U_l by layer, intact vs circuit-only; (c) where the three largest-writing heads at the arrow attend, by layer."""
import os,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D=os.environ.get('ICL_DATA','../data')
if not os.path.exists(f'{D}/trace/000_none.npz'): print('fig19: no trace shards under',D,'(run run.py with trace=1 intact, then with circ=<heads>; see README); skipped'); raise SystemExit(0)
z=np.load(f'{D}/trace/000_none.npz',allow_pickle=True); c=np.load(f'{D}/trace/000_circ_only.npz',allow_pickle=True)
T=int(z['T']); toks=[str(t) for t in z['tokstr']]; spans=z['spans']; L=z['wnorm'].shape[0]; late=list(z['late']); q=T-1
slot=sorted({p for a,b in spans for p in range(a,b)}); arrows=sorted({a-1 for a,b in spans}); qs=list(range(max(b for a,b in spans),T-1))
W=z['wnorm'].astype(np.float32).sum(-1)[:, :T]            # [L,T] total head write norm per position
fig,axs=plt.subplots(1,3,figsize=(7.2,3.3),gridspec_kw=dict(width_ratios=[2.0,1.0,1.5]))
ax=axs[0]; im=ax.imshow(np.log10(W+1e-3),aspect='auto',cmap='magma',origin='upper'); ax.set_xlabel('position'); ax.set_ylabel('layer'); ax.set_title('(a) head write norm, log10',fontsize=8,loc='left')
for p in slot: ax.plot([p],[-1.3],marker='v',color='k',ms=2.5,clip_on=False)
ax.plot([q],[-1.3],marker='v',color='#00b0c8',ms=3.5,clip_on=False); ax.text(q,-2.6,'query arrow',color='#00b0c8',fontsize=6,ha='right',clip_on=False); ax.text(spans[0][0],-2.6,'label slots',color='k',fontsize=6,clip_on=False)
plt.colorbar(im,ax=ax,fraction=0.03,pad=0.02)
ax=axs[1]; g=z['proj_g'][:,q]; gc=c['proj_g'][:,q]; ax.plot(g,range(L),color='#2ca02c',lw=1.2,label='g, intact'); ax.plot(gc,range(L),color='#2ca02c',lw=1.2,ls='--',label='g, circuit-only')
u=np.full(L,np.nan); uc=np.full(L,np.nan)
for i,l in enumerate(late): u[l]=z['proj_u'][i,q]; uc[l]=c['proj_u'][i,q]
ax2=ax.twiny(); ax2.plot(u,range(L),color='#1f77b4',lw=1.0,label='U, intact'); ax2.plot(uc,range(L),color='#1f77b4',lw=1.0,ls='--',label='U, circuit-only'); ax2.set_xlabel('U projection',fontsize=7,color='#1f77b4'); ax2.tick_params(labelsize=6)
ax.set_ylim(L-0.5,-0.5); ax.set_xlabel('g projection',fontsize=7,color='#2ca02c'); ax.set_ylabel('layer'); ax.tick_params(labelsize=6); ax.set_title('(b) arrow residual',fontsize=8,loc='left')
h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels(); ax.legend(h1+h2,l1+l2,fontsize=5.5,loc='upper center',bbox_to_anchor=(0.5,-0.22),ncol=2,frameon=False,columnspacing=0.8,handlelength=1.6)
ax=axs[2]; att=z['att']; cats={'matching slots':'#d62728','other slots':'#ff9896','sink (pos 0)':'#7f7f7f','arrows':'#9467bd','query sentence':'#8c564b','self':'#bcbd22','rest':'#dddddd'}
tgt=str(z['target']).strip(); match=[p for a,b in spans for p in range(a,b) if toks[a].strip()==tgt]
M=np.zeros((L,len(cats)))
for l in range(L):
    hh=np.argsort(-z['wnorm'][l,q].astype(np.float32))[:3]
    for h in hh:
        row=att[l,h,q,:q+1].astype(np.float32)
        for p,w in enumerate(row):
            k='sink (pos 0)' if p==0 else 'self' if p==q else 'matching slots' if p in match else 'other slots' if p in slot else 'arrows' if p in arrows else 'query sentence' if p in qs else 'rest'
            M[l,list(cats).index(k)]+=w/3
left=np.zeros(L)
for i,(k,col) in enumerate(cats.items()): ax.barh(range(L),M[:,i],left=left,color=col,height=0.9,label=k); left+=M[:,i]
ax.set_ylim(L-0.5,-0.5); ax.set_xlim(0,1); ax.set_xlabel('attention share (3 largest-writing heads)',fontsize=6.5); ax.tick_params(labelsize=6); ax.set_title('(c) what they read',fontsize=8,loc='left'); ax.legend(fontsize=5.5,loc='upper left',bbox_to_anchor=(1.01,1.0),frameon=False,handlelength=1.2)
for a in axs: 
    for sp in ['top','right']: a.spines[sp].set_visible(False)
fig.tight_layout(w_pad=0.8,rect=(0,0.02,1,1)); out=os.environ.get('ICL_FIGOUT','figures/fig19_trace.pdf'); fig.savefig(out); print('fig19 written',out, 'T',T,'match slots',match[:4])
