import numpy as np,matplotlib.pyplot as plt,figstyle as fs
fs.use()
fig,axs=plt.subplots(1,2,figsize=(7.0,2.5))
# (a) selection-draw interaction (notebook D110/D115 battery, intact +2.46) and fresh-draw confirmation (D141a/b, intact +0.84)
ax=axs[0]
rows=[('readers off',-1.96,-2.39,-1.53,fs.INTERV),('brakes off',4.88,3.52,6.24,fs.SECOND),('both off',-0.09,-0.89,0.71,fs.INTACT),('readers off (fresh draw)',-0.74,-0.95,-0.55,fs.INTERV),('brakes off (fresh draw)',1.39,0.88,1.88,fs.SECOND),('24 random (fresh draw)',0.22,0.14,0.31,fs.CONTROL),('20 random (fresh draw)',0.02,-0.03,0.06,fs.CONTROL)]
fs.effect_bar(ax,[r[0] for r in rows],[r[1] for r in rows],[r[2] for r in rows],[r[3] for r in rows],[r[4] for r in rows],xlabel='change in margin (nats)'); ax.set_title('readers × brakes',fontsize=9)
# (b) order term: last-position matching demo relative to earlier (nats)
ax=axs[1]; x=np.arange(3); intact=[-0.90,-0.05,-0.76]; lo_i=[-1.03,-0.53,-0.89]; hi_i=[-0.78,0.38,-0.62]; knocked=[0.16,1.08,np.nan]; lo_k=[-0.03,0.43,np.nan]; hi_k=[0.35,1.71,np.nan]
ax.bar(x-0.19,intact,width=0.36,color=fs.INTACT,label='intact',yerr=[np.array(intact)-np.array(lo_i),np.array(hi_i)-np.array(intact)],error_kw=dict(lw=0.7,capsize=1.5))
ax.bar(x+0.19,knocked,width=0.36,color=fs.SECOND,label='brakes off',yerr=[np.nan_to_num(np.array(knocked)-np.array(lo_k)),np.nan_to_num(np.array(hi_k)-np.array(knocked))],error_kw=dict(lw=0.7,capsize=1.5))
ax.axhline(0,color=fs.INTACT,lw=0.6); ax.set_xticks(x); ax.set_xticklabels(['Qwen\n6 demos','Qwen\n12 demos','SmolLM3\n6 demos'],fontsize=7.5); ax.set_ylabel('order term (nats)'); ax.legend(fontsize=7,loc='upper left'); ax.set_title('recency: the brakes set the sign',fontsize=9)
for ax,s in zip(axs,'ab'): fs.panel_label(ax,s,x=-0.55 if s=='a' else -0.2)
fig.subplots_adjust(wspace=0.75); fig.savefig('figures/fig6_populations.pdf'); fig.savefig('figures/fig6_populations.png'); print('fig6 written')
