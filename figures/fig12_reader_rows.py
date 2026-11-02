"""Fig 12: attention rows of three readers (Qwen3-4B) on recurring-label prompts, with the 16 repeat detectors knocked, and on unique-label prompts (D145): mass on the sink vs the four label slots. Mean over 30 prompts of the slotgate battery."""
import os
RUNS=os.environ.get("ICL_RUNS","runs"); DATA=os.environ.get("ICL_DATA","data")
import json,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, figstyle as fs
rows=[json.loads(l) for l in open(f'{DATA}/D145_reader_rows.jsonl')]  # subset of the run's knock.jsonl (fields used here)
heads=['22:20','24:21','29:11']; conds=[('icl_slotgate_two','none','labels repeat'),('icl_slotgate_two','lookback16','repeat detectors knocked'),('icl_slotgate_one','none','unique labels')]
def mass(r,h):
    sp=[a for a,b in r['spans']]; d={p:m for p,_,m in r['arow'][h]['top']}
    return [d.get(0,0)]+[d.get(p,0) for p in sp]
M=np.zeros((len(heads),len(conds),5))
for i,h in enumerate(heads):
    for j,(k,c,_) in enumerate(conds):
        sel=[r for r in rows if r['kind']==k and r['cond']==c and h in r['arow']]
        M[i,j]=np.mean([mass(r,h) for r in sel],axis=0)
fig,axs=plt.subplots(1,3,figsize=(7.2,2.3))
for i,ax in enumerate(axs):
    im=ax.imshow(M[i],vmin=0,vmax=1,cmap='Greys',aspect='auto')
    ax.set_xticks(range(5)); ax.set_xticklabels(['sink','slot 1','slot 2','slot 3','slot 4'],fontsize=7,rotation=30)
    ax.set_yticks(range(3)); ax.set_yticklabels([n for *_,n in conds] if i==0 else [],fontsize=7)
    ax.set_title(f'reader {heads[i]}',fontsize=9)
    for j in range(3):
        for k in range(5): ax.text(k,j,f'{M[i,j,k]:.2f}',ha='center',va='center',fontsize=6,color='w' if M[i,j,k]>0.5 else 'k')
fig.colorbar(im,ax=axs,fraction=0.02,pad=0.02,label='attention mass')
fig.savefig('figures/fig12_reader_rows.pdf',bbox_inches='tight'); print('fig12 written'); print(np.round(M,2))
