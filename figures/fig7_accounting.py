import json,numpy as np,collections,matplotlib.pyplot as plt,figstyle as fs
import os
RUNS=os.environ.get("ICL_RUNS","runs"); DATA=os.environ.get("ICL_DATA","data")
fs.use()
def load(d,nl):
    R=collections.defaultdict(dict)
    for l in open(d):
        r=json.loads(l)
        if r.get('dla') is None or r.get('alt_tlp') is None or (r['cond'].startswith('tp_') and 'hout' in r['cond']): continue
        R[r['cond']][(r['seq'],r['kind'])]=r
    return R
def bands(r,nl):
    A=np.zeros(nl); M=np.zeros(nl)
    f=r['frms']
    for l,v in r['dla'].items(): A[int(l)]=sum(v)/f
    for l,v in r['mdla'].items(): M[int(l)]=v['tot']/f
    cut=(21,30) if nl==36 else (21,30)
    return dict(att_early=A[:cut[0]].sum(),att_read=A[cut[0]:cut[1]].sum(),att_late=A[cut[1]:].sum(),mlp_early=M[:cut[0]].sum(),mlp_read=M[cut[0]:cut[1]].sum(),mlp_late=M[cut[1]:].sum(),emb=r['dlae']/r['frms'] if abs(r['dlae'])>abs(r['dlafin'])*5 else r['dlae'],fin=r['dlafin'],rec=r['dlarec']),A
runs={'Qwen3-4B':f'{DATA}/D159_dla_Qwen3-4B.jsonl','SmolLM3-3B':f'{DATA}/D159_dla_SmolLM3-3B.jsonl'}  # subset of the run's knock.jsonl (fields used here)
keys=['att_early','att_read','att_late','mlp_early','mlp_read','mlp_late','emb']; names=['attn 0–20','attn 21–29','attn 30–35','MLP 0–20','MLP 21–29','MLP 30–35','embed']
fig,axs=plt.subplots(1,3,figsize=(7.0,2.5),gridspec_kw={'width_ratios':[1.2,1.2,1]})
conc={}
for ax,(fam,d) in zip(axs[:2],runs.items()):
    R=load(d,36); nl=36
    base=R['none']; flip=R['tp_qx_exact']
    B=[bands(r,nl) for r in base.values()]; F=[(bands(flip[k],nl)[0],bands(base[k],nl)[1],bands(flip[k],nl)[1]) for k in flip if k in base]
    mb=np.array([[b[0][k] for k in keys] for b in B]).mean(0); mf=np.array([[f[0][k] for k in keys] for f in F]).mean(0)
    err=np.mean([abs(b[0]['fin']-b[0]['rec']) for b in B]); print(fam,'n',len(B),'intact bands',dict(zip(keys,np.round(mb,2))),'flip change',dict(zip(keys,np.round(mf-mb,2))),'fin',np.mean([b[0]['fin'] for b in B]).round(2))
    y=np.arange(len(keys))[::-1]; ax.barh(y+0.18,mb,height=0.34,color=fs.INTACT,label='intact'); ax.barh(y-0.18,mf-mb,height=0.34,color=fs.INTERV,label='class flip (change)')
    ax.axvline(0,color=fs.INTACT,lw=0.6); ax.set_yticks(y); ax.set_yticklabels(names); ax.set_xlabel('direct logit (nats)'); ax.set_title(fam,fontsize=9)
    dh=np.array([(f[2]-f[1]) for f in F]).mean(0)   # per-layer attention change; concentration needs per-head: recompute
    perhead=collections.defaultdict(list)
    for k in flip:
        if k not in base: continue
        for l in flip[k]['dla']:
            for h,(a,b) in enumerate(zip(flip[k]['dla'][l],base[k]['dla'][l])): perhead[(int(l),h)].append(a/flip[k]['frms']-b/base[k]['frms'])
    ph=np.array([np.mean(v) for v in perhead.values()]); tot=ph[ph<0].sum(); srt=np.sort(ph[ph<0]); conc[fam]=np.cumsum(srt)/tot
axs[0].legend(loc='lower right',fontsize=7)
ax=axs[2]
for fam,c in conc.items(): ax.plot(np.arange(1,len(c)+1),c,color=fs.MODEL[fam],label=fam)
ax.axhline(0.5,color=fs.CONTROL,lw=0.6,ls='--'); ax.axhline(0.8,color=fs.CONTROL,lw=0.6,ls='--'); ax.set_xscale('log'); ax.set_xlabel('heads (ranked)'); ax.set_ylabel('share of direct flip'); ax.set_ylim(0,1.02); ax.legend(fontsize=7,loc='lower right')
for ax,s in zip(axs,'abc'): fs.panel_label(ax,s,x=-0.28 if s!='c' else -0.3)
for fam,c in conc.items(): print(fam,'50% at',int(np.searchsorted(c,0.5))+1,'80% at',int(np.searchsorted(c,0.8))+1)
fig.subplots_adjust(wspace=0.55); fig.savefig('figures/fig7_accounting.pdf'); fig.savefig('figures/fig7_accounting.png'); print('fig7 written')
