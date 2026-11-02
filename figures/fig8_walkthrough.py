"""Fig 8: one 12-shot prompt followed through Qwen3-4B: which tokens are acted on at which layers by which stage."""
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np, figstyle as fs
toks=['s1','->','foo','s2','->','bar','...','s12','->','quil','q','->']
LAB=[2,5,9]; ARR=[1,4,8]; Q=11
# name, layer range, columns, x-offset, width, colour
stages=[('duplicate detectors',(1,9),LAB,0,0.9,'#999999'),
 ('encoder',(4,13),ARR+[Q],0,0.9,'#555555'),
 ('writers',(10,23),LAB,0,0.9,fs.MODEL['Qwen3-4B']),
 ('gate g',(18,23),[Q],-0.33,0.28,'#2ca02c'),
 ('query code',(22,23),[Q],0.0,0.28,'#8c564b'),
 ('readers',(21,29),[Q],0.33,0.28,fs.INTERV),
 ('brakes',(22,34),[Q],-0.33,0.28,'#9467bd'),
 ('readout',(30,35),[Q],0.17,0.6,'#1f77b4')]
fig,ax=plt.subplots(figsize=(7.2,4.4))
for name,(l0,l1),cols,dx,w,c in stages:
    for x in cols: ax.add_patch(plt.Rectangle((x+dx-w/2,l0-0.4),w,l1-l0+0.8,color=c,alpha=0.85,lw=0))
# legend-like labels at fixed, non-overlapping heights on the right
ys={'duplicate detectors':3,'encoder':8.5,'writers':16,'gate g':20.5,'query code':23,'readers':25.5,'brakes':28.5,'readout':32.5}
for name,(l0,l1),cols,dx,w,c in stages:
    ax.annotate(name,xy=(Q+dx+w/2 if Q in cols else cols[-1]+0.45,(l0+l1)/2),xytext=(len(toks)+0.3,ys[name]),fontsize=8,color=c,va='center',
                arrowprops=dict(arrowstyle='-',color=c,lw=0.6,alpha=0.7))
ax.set_xticks(range(len(toks))); ax.set_xticklabels(toks,fontsize=8); ax.set_xlim(-0.6,len(toks)+3.2)
ax.set_ylim(36,-0.5); ax.set_ylabel('layer'); ax.set_xlabel('token')
for sp in ['top','right']: ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig('figures/fig8_walkthrough.pdf'); print('fig8 written')
