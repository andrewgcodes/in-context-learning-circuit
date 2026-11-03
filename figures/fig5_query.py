import numpy as np,matplotlib.pyplot as plt,figstyle as fs
fs.use()
fig,axs=plt.subplots(1,2,figsize=(7.0,2.4),gridspec_kw={'width_ratios':[1.3,1]})
# (a) D151 concept12 n=144: pair handoff with holds (effects on margin, nats)
rows=[('pair handoff',-0.528,-0.681,-0.386,fs.INTERV),('hold MLPs 22–35',-0.368,-0.468,-0.274,fs.CONTROL),('hold MLPs 22–29',-0.290,-0.372,-0.204,fs.CONTROL),('hold attn 24–29',-0.136,-0.174,-0.099,fs.PRED),('hold attn 24–35',-0.044,-0.059,-0.030,fs.PRED),('hold attn + MLPs',-0.007,-0.018,0.003,fs.PRED),('hold MLPs 30–35',-0.662,-0.834,-0.500,fs.SECOND)]
fs.effect_bar(axs[0],[r[0] for r in rows],[r[1] for r in rows],[r[2] for r in rows],[r[3] for r in rows],[r[4] for r in rows],xlabel='effect of pair handoff (nats)'); axs[0].set_title('who consumes the code',fontsize=9)
# (b) subspace shares: D153 (13 carriers), D154/D155 (89 readers), random 89-dim; 12-shot and 4-demo
ax=axs[1]; x=np.arange(3); s12=[0.19,0.64,0.04]; s4=[np.nan,0.50,0.00]
ax.bar(x-0.19,s12,width=0.36,color=fs.INTERV,label='12 demos'); ax.bar(x+0.19,s4,width=0.36,color='#F08080',label='4 demos')
ax.set_xticks(x); ax.set_xticklabels(['13 carrier\nfunctionals','89 reader\nfunctionals','random\n89-dim'],fontsize=7.5); ax.set_ylabel('share of pair effect'); ax.set_ylim(0,0.8); ax.legend(fontsize=7); ax.set_title('what the code is',fontsize=9)
for ax,s in zip(axs,'ab'): fs.panel_label(ax,s,x=-0.55 if s=='a' else -0.25)
fig.subplots_adjust(wspace=0.7); fig.savefig('figures/fig5_query.pdf'); fig.savefig('figures/fig5_query.png'); print('fig5 written')
