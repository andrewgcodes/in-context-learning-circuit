import re,json,numpy as np,matplotlib.pyplot as plt,figstyle as fs
import os
RUNS=os.environ.get("ICL_RUNS","runs"); DATA=os.environ.get("ICL_DATA","data")
fs.use()
def parse(f):
    H={}
    for l in open(f):
        m=re.match(r'(\d+:\d+)\s+L=([+-]\d+\.\d+) Ls=([+-]\d+\.\d+) nrest=(\d+) fit: (.*?) maxerr',l)
        if not m: continue
        fit=[(int(k),float(a),float(b)) for k,a,b in re.findall(r'K(\d) ([\d.]+)/([\d.]+)',m.group(5))]
        oos=re.search(r'OOS 1-slot: (.*?) maxerr',l); oosv=[(int(k),float(a),float(b)) for k,a,b in re.findall(r'K(\d) ([\d.]+)/([\d.]+)',oos.group(1))] if oos else []
        H[m.group(1)]=dict(L=float(m.group(2)),fit=fit,oos=oosv)
    return H
Q=parse(f'{DATA}/D139_law_qwen.txt'); L=parse(f'{DATA}/D161_law_llama.txt')
D162={'Qwen3-4B':json.load(open(f'{DATA}/D162_Qwen3-4B_pred_vs_meas.json')),'Llama-3.2-3B':json.load(open(f'{DATA}/D162_Llama-3.2-3B_pred_vs_meas.json'))}
fig,axs=plt.subplots(1,3,figsize=(7.0,2.5))
def sc(ax,pairs,color,label,ms=7,alpha=0.55):
    p=np.array(pairs); ax.scatter(p[:,1],p[:,0],s=ms,color=color,alpha=alpha,lw=0,label=label)
for ax in axs: ax.plot([0,1],[0,1],color=fs.CONTROL,lw=0.8,zorder=0); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xlabel('predicted mass'); ax.set_aspect('equal')
sc(axs[0],[(a,b) for h in Q.values() for _,a,b in h['fit']],fs.MODEL['Qwen3-4B'],'Qwen, fitted'); axs[0].set_ylabel('measured mass'); axs[0].set_title('12 demos (fitted)')
sc(axs[1],[(a,b) for h in Q.values() for _,a,b in h['oos']],fs.MODEL['Qwen3-4B'],'Qwen'); sc(axs[1],[(a,b) for h in L.values() for _,a,b in h['oos']],fs.MODEL['Llama-3.2-3B'],'Llama'); axs[1].set_title('1 slot per class (held out)')
for fam,c in (('Qwen3-4B',fs.MODEL['Qwen3-4B']),('Llama-3.2-3B',fs.MODEL['Llama-3.2-3B'])):
    sc(axs[2],[(m,p) for h in D162[fam]['heads'] for _,m,p in h['K']],c,fam.split('-')[0] if fam.startswith('L') else 'Qwen')
axs[2].set_title('18 demos (held out, failed)'); axs[1].legend(loc='upper left',handletextpad=0.2,markerscale=1.6); 
for ax,s in zip(axs,'abc'): fs.panel_label(ax,s,x=-0.2)
for ax in axs[1:]: ax.set_yticklabels([])
fig.subplots_adjust(wspace=0.15); fig.savefig('figures/fig3_law.pdf'); fig.savefig('figures/fig3_law.png'); print('fig3 written', len(Q), len(L))
