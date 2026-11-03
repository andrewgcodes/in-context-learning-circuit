"""Fig 21: natural-label closure. For each model, R (follows demonstrations) and P (follows word meaning) under intact, circuit-only, circuit-only resampled, random set resampled, no heads; parsed from analysis/*_decomp.txt (analyze_D185_decomp.py output).
usage: python3 fig21_closure.py [out.pdf Qwen=path/D186Q2nK2_decomp.txt SmolLM3=... Llama=...]; with no arguments reads the three decomp files from $ICL_DATA (default ../data) and writes figures/fig21_closure.pdf"""
import os,sys,re,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
def parse(fn):
    d={}; 
    for ln in open(fn):
        m=re.match(r'intact: R \(follows demos\) ([+-][\d.]+) \[([+-][\d.]+),([+-][\d.]+)\]\s+P \(follows meaning\) ([+-][\d.]+) \[([+-][\d.]+),([+-][\d.]+)\]',ln)
        if m: d['intact']=tuple(map(float,m.groups())); continue
        m=re.match(r'(\S+)\s+([+-][\d.]+) \[([+-][\d.]+),([+-][\d.]+)\]\s+([+-][\d.]+) \[([+-][\d.]+),([+-][\d.]+)\]',ln)
        if m: d[m.group(1)]=tuple(map(float,m.groups()[1:]))
    return d
D=os.environ.get('ICL_DATA','../data')
if len(sys.argv)>1: out=sys.argv[1]; models=[a.split('=',1) for a in sys.argv[2:]]
else: out='figures/fig21_closure.pdf'; models=[('Qwen',f'{D}/D186Q2nK2_decomp.txt'),('SmolLM3',f'{D}/D186SQ2nK2_decomp.txt'),('Llama',f'{D}/D186LQ2nK_decomp.txt')]
conds=[('intact','intact'),('circ_only','circuit only'),('circ_only_rs','circuit only\n(resampled)'),('circ_rand_only_rs','random set\n(resampled)'),('all_heads_knock_rs','no heads\n(resampled)')]
fig,axs=plt.subplots(1,len(models),figsize=(3.6*len(models),3.2),sharey=True)
for ax,(name,fn) in zip(np.atleast_1d(axs),models):
    d=parse(fn); R0,Rl,Rh,P0,Pl,Ph=d['intact']; x=np.arange(len(conds))
    R=[];Rc=[];P=[];Pc=[]
    for k,_ in conds:
        if k=='intact': R.append(R0);Rc.append((R0-Rl,Rh-R0));P.append(P0);Pc.append((P0-Pl,Ph-P0))
        else:
            v=d.get(k); 
            if v is None: R.append(np.nan);Rc.append((0,0));P.append(np.nan);Pc.append((0,0)); continue
            dR,dRl,dRh,dP,dPl,dPh=v; R.append(R0+dR);Rc.append((dR-dRl,dRh-dR));P.append(P0+dP);Pc.append((dP-dPl,dPh-dP))
    ax.bar(x-0.2,R,0.38,yerr=np.array(Rc).T,color='#1f5fa8',capsize=2,label='follows demonstrations (R)')
    ax.bar(x+0.2,P,0.38,yerr=np.array(Pc).T,color='#d9822b',capsize=2,label='follows word meaning (P)')
    ax.axhline(0,color='k',lw=0.6); ax.set_xticks(x); ax.set_xticklabels([c[1] for c in conds],fontsize=7); ax.set_title(name,fontsize=10)
    for s in ('top','right'): ax.spines[s].set_visible(False)
np.atleast_1d(axs)[0].set_ylabel('nats'); np.atleast_1d(axs)[0].legend(fontsize=7,frameon=False,loc='upper right')
plt.tight_layout(); plt.savefig(out); print('wrote',out)
