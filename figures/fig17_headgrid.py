import os
"""Layer x head map of single-head knock effects on the 12-shot class margin (paired, 144 prompts x 6 label families): Qwen3-4B layers 21-35, Llama-3.2-3B layers 16-27. Data: data_headgrid.json (D98 / D161cen3)."""
import json,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.patches as mp
G=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'data_headgrid.json')))
fig,axes=plt.subplots(1,2,figsize=(7.2,3.1),gridspec_kw={'width_ratios':[32,24]})
for ax,(tag,name) in zip(axes,[('qwen','Qwen3-4B-Base'),('llama','Llama-3.2-3B')]):
    g=G[tag]; L=sorted(set(int(c.split('_')[1]) for c in g)); H=sorted(set(int(c.split('_')[2]) for c in g))
    M=np.full((len(L),len(H)),np.nan)
    for c,v in g.items(): _,l,h=c.split('_'); M[L.index(int(l)),H.index(int(h))]=v
    v=0.15; im=ax.imshow(M,cmap='RdBu',vmin=-v,vmax=v,aspect='auto',origin='lower')
    ax.set_xticks(range(0,len(H),4)); ax.set_xticklabels(H[::4],fontsize=6); ax.set_yticks(range(len(L))); ax.set_yticklabels(L,fontsize=6)
    ax.set_xlabel('head',fontsize=7); ax.set_ylabel('layer',fontsize=7); ax.set_title(name,fontsize=8)
    flat=sorted(g.items(),key=lambda kv:kv[1])
    for c,_ in flat[:5]+flat[-5:]:
        _,l,h=c.split('_'); col='#c0303a' if g[c]<0 else '#2a3fa0'
        ax.add_patch(mp.Rectangle((H.index(int(h))-0.5,L.index(int(l))-0.5),1,1,fill=False,ec=col,lw=1.1))
    ax.tick_params(length=2)
cb=fig.colorbar(im,ax=axes,fraction=0.025,pad=0.02); cb.set_label('knock effect on class margin (nats)',fontsize=7); cb.ax.tick_params(labelsize=6)
fig.savefig('figures/fig17_headgrid.pdf',bbox_inches='tight'); fig.savefig('/tmp/fig17.png',dpi=150,bbox_inches='tight'); print('fig17 written')
