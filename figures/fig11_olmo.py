"""Fig 11: emergence order across OLMo-2 stage-1 checkpoints (D148 1B, D150 7B): copy, then null action, then class-keyed retrieval."""
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np, figstyle as fs
tok1=[21,105,315,839,1887,4000]; tok7=[42,210,630,1680,3900]
fig,axs=plt.subplots(1,3,figsize=(7.2,2.5),sharex=True)
c1,c7='#999999','#111111'
ax=axs[0]; ax.plot(tok1,[1,1,1,1,1,1],'o-',color=c1,label='1B'); ax.plot(tok7,[1,1,1,1,1],'s-',color=c7,label='7B'); ax.set_ylim(-0.05,1.1); ax.set_title('copy a repeated label',fontsize=9); ax.set_ylabel('label-argmax rate\n(recurring prompts)',fontsize=7); ax.legend(fontsize=7,frameon=False)
ax=axs[1]; ax.plot(tok1,[.63,.73,.60,.13,.03,.17],'o-',color=c1); ax.plot(tok7,[.17,.33,.17,.20,.00],'s-',color=c7); ax.set_ylim(-0.05,1.1); ax.set_title('null action',fontsize=9); ax.set_ylabel('label-argmax rate\n(unique-label prompts)',fontsize=7)
ax=axs[2]; ax.plot(tok1,[-3.5,-2.5,-1.5,-1.0,-0.7,-0.5],'o--',color=c1,alpha=0.5); ax.plot(tok7,[-0.94,-0.70,0.86,-0.15,0.72],'s-',color=c7); ax.axhline(0,color='k',lw=0.6); ax.set_title('class-keyed retrieval',fontsize=9); ax.set_ylabel('unique-prompt margin (nats)',fontsize=7)
for ax in axs:
    ax.set_xscale('log'); ax.set_xlabel('training tokens (B)',fontsize=8)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig('figures/fig11_olmo.pdf'); print('fig11 written')
