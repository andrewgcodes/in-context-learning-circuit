"""Fig 14: the key is written in the middle layers and the write must be spread across depth. (a) Llama: sentence-exchange effect carried by label-token K/V handed per layer band. (b) SmolLM3: adding the class direction u to the label tokens, same total dose, in one layer vs spread over four (D138s, 144 non-binding-axis prompts, key + query push)."""
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np, figstyle as fs
fig,axs=plt.subplots(1,2,figsize=(7.2,2.5))
ax=axs[0]; bands=['L0–6','L7–13','L14–20','L21–27']; v=[-0.02,-0.15,-0.15,-0.04]
ax.bar(range(4),[x/-0.23 for x in v],color=[ '#bbbbbb',fs.MODEL['Llama-3.2-3B'],fs.MODEL['Llama-3.2-3B'],'#bbbbbb'])
ax.set_xticks(range(4)); ax.set_xticklabels(bands,fontsize=8); ax.set_ylabel('share of sentence-exchange effect',fontsize=7); ax.set_title('Llama: the key lives in the middle layers',fontsize=9)
for i,x in enumerate(v): ax.text(i,x/-0.23+0.02,f'{x:+.2f}',ha='center',fontsize=7)
ax=axs[1]; x=np.array([1,2]); one=[0.37,0.20]; four=[0.56,0.69]; e1=[[0.08,0.08],[0.09,0.09]]; e4=[[0.08,0.08],[0.09,0.09]]
ax.errorbar(x-0.05,one,yerr=[[0.08,0.11],[0.09,0.11]],fmt='o-',color='#999999',capsize=2,label='one layer'); ax.errorbar(x+0.05,four,yerr=[[0.08,0.11],[0.09,0.12]],fmt='s-',color=fs.INTERV,capsize=2,label='spread over four layers')
ax.set_xticks([1,2]); ax.set_xticklabels(['natural dose','double dose'],fontsize=8); ax.set_ylabel('margin gain (nats)',fontsize=8); ax.set_title('SmolLM3: a spread write helps more',fontsize=9); ax.legend(fontsize=7,frameon=False,loc='upper left'); ax.set_xlim(0.6,2.4)
for ax in axs:
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig('figures/fig14_writers.pdf'); print('fig14 written')
