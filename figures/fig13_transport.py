"""Fig 13: value transport. Share of the label-exchange swing (cvc_sym, 12-shot, Qwen) carried by handing the VALUES of layer 21-29 KV groups at the label tokens: per layer and per single group. Numbers from D143."""
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np, figstyle as fs
tot=2.33
fig,axs=plt.subplots(1,2,figsize=(7.2,2.5),gridspec_kw={'width_ratios':[1,1.3]})
ax=axs[0]; names=['whole twin','all slot K+V','all values\n(72 groups)']; v=[2.33,1.82,2.00]
ax.barh(range(3),[x/tot for x in v],color=[fs.INTACT,fs.INTERV,fs.INTERV]); ax.set_yticks(range(3)); ax.set_yticklabels(names,fontsize=7); ax.invert_yaxis(); ax.set_xlabel('share of label-exchange swing',fontsize=8); ax.set_xlim(0,1.05); ax.set_title('where the swing lives',fontsize=9)
for i,x in enumerate(v): ax.text(x/tot+0.01,i,f'{x/tot:.2f}',va='center',fontsize=7)
ax=axs[1]; layers=['L22','L24','L29']; lv=[0.31,0.69,0.64]
ax.bar(np.arange(3)-0.2,[x/tot for x in lv],0.4,color='#999999',label='whole layer (values)')
groups=['22.5','24.5','29.2']; gv=[0.31,0.59,0.38]; pos={'L22':0,'L24':1,'L29':2}
ax.bar([pos[f'L{g.split(".")[0]}']+0.2 for g in groups],[x/tot for x in gv],0.4,color=fs.INTERV,label='single KV group')
ax.axhline(0.03/tot,color='k',lw=0.6,ls=':'); ax.text(2.45,0.03/tot+0.01,'random',fontsize=6)
ax.set_xticks(range(3)); ax.set_xticklabels(layers,fontsize=8); ax.set_ylabel('share of swing',fontsize=8); ax.set_title('three KV groups carry half',fontsize=9); ax.legend(fontsize=6,frameon=False)
for ax in axs:
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig('figures/fig13_transport.pdf'); print('fig13 written')
