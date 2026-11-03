"""Fig 22: Qwen3-4B base vs instruct on word-label prompts, 189 base heads in both. R (follows demonstrations) and P (follows word meaning) under intact, circuit only (mean reference), circuit only (resampled), random same-size set (resampled), no heads (resampled); parsed from analyze_D185_decomp.py output.
usage: python3 fig22_instruct.py  (reads $ICL_DATA/D186Q2nK2_decomp.txt and D187QIn_decomp.txt; ICL_DATA defaults to ../data)"""
import os,re,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, figstyle as fs
fs.use(); A=os.environ.get('ICL_DATA','../data')
def parse(fn):
    d={}
    for ln in open(fn):
        m=re.match(r'intact: R \(follows demos\) ([+-][\d.]+) \[([+-][\d.]+),([+-][\d.]+)\]\s+P \(follows meaning\) ([+-][\d.]+) \[([+-][\d.]+),([+-][\d.]+)\]',ln)
        if m: d['intact']=tuple(map(float,m.groups())); continue
        m=re.match(r'(\S+)\s+([+-][\d.]+) \[([+-][\d.]+),([+-][\d.]+)\]\s+([+-][\d.]+) \[([+-][\d.]+),([+-][\d.]+)\]',ln)
        if m: d[m.group(1)]=tuple(map(float,m.groups()[1:]))
    return d
models=[('Qwen3-4B base',f'{A}/D186Q2nK2_decomp.txt'),('Qwen3-4B instruct',f'{A}/D187QIn_decomp.txt')]
conds=[('intact','intact'),('circ_only','189 heads\nalone'),('circ_only_rs','189 heads\nalone, resampled'),('circ_rand_only_rs','random 189\nresampled'),('all_heads_knock_rs','no heads')]
fig,axs=plt.subplots(1,2,figsize=(7.2,2.6),sharey=True)
for ax,(name,fn),lab in zip(axs,models,'ab'):
    d=parse(fn); R0,Rl,Rh,P0,Pl,Ph=d['intact']; x=np.arange(len(conds)); R=[];Rc=[];P=[];Pc=[]
    for k,_ in conds:
        if k=='intact': R.append(R0);Rc.append((R0-Rl,Rh-R0));P.append(P0);Pc.append((P0-Pl,Ph-P0))
        else:
            dR,dRl,dRh,dP,dPl,dPh=d[k]; R.append(R0+dR);Rc.append((dR-dRl,dRh-dR));P.append(P0+dP);Pc.append((dP-dPl,dPh-dP))
    ek=dict(lw=0.8,capsize=2,ecolor=fs.INTACT)
    ax.bar(x-0.19,R,0.36,yerr=np.array(Rc).T,color=fs.PRED,error_kw=ek,label='follows the demonstrations ($R$)')
    ax.bar(x+0.19,P,0.36,yerr=np.array(Pc).T,color=fs.SECOND,error_kw=ek,label='follows the words\' meaning ($P$)')
    ax.axhline(0,color=fs.INTACT,lw=0.7); ax.set_xticks(x); ax.set_xticklabels([c[1] for c in conds],fontsize=7); ax.set_title(name); fs.panel_label(ax,lab)
    ax.axhspan(Rl,Rh,color=fs.PRED,alpha=0.08,lw=0)
axs[0].set_ylabel('part of the word-label margin (nats)'); axs[0].legend(loc='upper left',handlelength=1.2)
axs[0].text(0.98,0.97,'band: intact $R$ interval',transform=axs[0].transAxes,ha='right',va='top',fontsize=7,color=fs.CONTROL)
plt.tight_layout(w_pad=1.5); plt.savefig('figures/fig22_instruct.pdf'); print('saved figures/fig22_instruct.pdf')
