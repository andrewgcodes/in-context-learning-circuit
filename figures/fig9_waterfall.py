"""Fig 9: waterfall of the direct-logit accounting (Qwen, 12-shot): intact margin and the class flip, by layer band. Numbers from the D159 exact accounting (Section: Where every nat...)."""
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np, figstyle as fs
def waterfall(ax,steps,title):
    x=0; run=0
    for name,v in steps:
        c=fs.INTERV if v<0 else fs.INTACT
        ax.bar(x,v,bottom=run,color=c,width=0.6); ax.text(x,run+v+(0.03 if v>=0 else -0.03),f'{v:+.2f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=7)
        run+=v; x+=1
    ax.bar(x,run,color='#999999',width=0.6); ax.text(x,run+(0.03 if run>=0 else -0.03),f'{run:+.2f}',ha='center',va='bottom' if run>=0 else 'top',fontsize=7)
    ax.set_xticks(range(x+1)); ax.set_xticklabels([n for n,_ in steps]+['total'],fontsize=7,rotation=20); ax.axhline(0,color='k',lw=0.6)
    ax.set_title(title,fontsize=9); ax.set_ylabel('nats')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
fig,axs=plt.subplots(1,2,figsize=(7.2,2.8))
waterfall(axs[0],[('embed + L0–20',-0.01),('attn 21–29',0.14),('attn 30–35',0.66),('late MLPs',-0.11)],'intact margin'); axs[0].set_ylim(-0.08,0.98)
waterfall(axs[1],[('pre-21 routes',-0.10),('attn 21–29',-0.36),('attn 30–35',-1.12),('late MLPs',0.25)],'change under class flip'); axs[1].set_ylim(-1.8,0.35)
fig.tight_layout(); fig.savefig('figures/fig9_waterfall.pdf'); print('fig9 written')
