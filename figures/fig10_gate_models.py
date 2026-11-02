"""Fig 10: the null action in Qwen vs Llama. (a) share of unique-label prompts that still receive a label. (b) Qwen: +g push lowers sink-minus-match at the three strongest readers. (c) Llama: knocking the 16 induction-scored heads moves reader mass from the sink to the slots."""
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np, figstyle as fs
fig,axs=plt.subplots(1,3,figsize=(7.2,2.6),gridspec_kw={'width_ratios':[0.8,1,1.2]})
ax=axs[0]; ax.bar([0,1],[10,67],color=[fs.MODEL['Qwen3-4B'],fs.MODEL['Llama-3.2-3B']],width=0.6)
ax.set_xticks([0,1]); ax.set_xticklabels(['Qwen','Llama'],fontsize=8); ax.set_ylabel('% unique-label prompts\nreceiving a label',fontsize=7); ax.set_title('gate strength',fontsize=9)
ax=axs[1]; heads=['24:8','28:0','29:11']; sm=[-1.75,-3.25,-3.65]; mm=[0.47,0.34,0.48]
x=np.arange(3); ax.bar(x-0.18,sm,0.36,color=fs.INTERV,label='sink − match'); ax.bar(x+0.18,mm,0.36,color='#999999',label='match − other')
ax.axhline(0,color='k',lw=0.6); ax.set_xticks(x); ax.set_xticklabels(heads,fontsize=8); ax.set_ylabel('logit change under +g',fontsize=7); ax.set_title('Qwen: +g opens the read',fontsize=9); ax.legend(fontsize=6,frameon=False)
ax=axs[2]; heads=['17:9','17:11','14:1','14:2','14:22']; sink=[-0.52,-0.40,-0.19,-0.17,-0.14]; slot=[0.39,0.29,0.30,0.26,0.22]
x=np.arange(5); ax.bar(x-0.18,sink,0.36,color='#777777',label='sink mass'); ax.bar(x+0.18,slot,0.36,color=fs.MODEL['Llama-3.2-3B'],label='slot mass')
ax.axhline(0,color='k',lw=0.6); ax.set_xticks(x); ax.set_xticklabels(heads,fontsize=7); ax.set_ylabel('mass change, induction heads knocked',fontsize=7); ax.set_title('Llama: knock releases readers',fontsize=9); ax.legend(fontsize=6,frameon=False)
for ax in axs:
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig('figures/fig10_gate_models.pdf'); print('fig10 written')
