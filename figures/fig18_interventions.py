"""The four interventions as schematics (IOI Fig. 3 idiom): a prompt column, the patched component in red, the measured output at END."""
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.patches as mp
fig,axes=plt.subplots(1,4,figsize=(7.2,2.6))
RED='#c0303a'; GREY='#777777'; BLK='#222222'
def column(ax,title,rows,hl=None):
    ax.set_xlim(0,10); ax.set_ylim(0.2,10); ax.axis('off'); ax.set_title(title,fontsize=7.5,pad=4)
    for i,(lab,txt) in enumerate(rows):
        y=9.0-i*1.25; fc='#ffe3e3' if hl==lab else '#f2f2f2'
        ax.add_patch(mp.FancyBboxPatch((0.3,y-0.45),2.0,0.9,boxstyle='round,pad=0.08',fc=fc,ec='none')); ax.text(1.3,y,lab,ha='center',va='center',fontsize=6,weight='bold',color=RED if hl==lab else BLK)
        ax.text(2.7,y,txt,ha='left',va='center',fontsize=6.3,family='monospace')
    return {lab:9.0-i*1.25 for i,(lab,_) in enumerate(rows)}
rows=[('S11','Lovely meal'),('L11','foo'),('S12','Went back'),('L12','bar'),('Q','Awful place'),('END','->')]
# (a) sentence swap: patch label-token keys from the twin prompt
Y=column(axes[0],'(a) sentence swap (keys)',rows,hl='L12')
axes[0].annotate('',xy=(7.2,Y['L12']),xytext=(7.2,Y['L12']+2.0),arrowprops=dict(arrowstyle='-|>',color=RED,lw=1.1)); axes[0].text(7.6,Y['L12']+1.0,'K from twin\n(S12 of the\nother class)',fontsize=5.6,color=RED,va='center')
axes[0].text(4.2,Y['END']-0.9,'read: margin at END',fontsize=5.6,color=GREY,va='center')
# (b) label swap: patch label-token values
Y=column(axes[1],'(b) label swap (values)',rows,hl='L12')
axes[1].annotate('',xy=(7.2,Y['L12']),xytext=(7.2,Y['L12']+2.0),arrowprops=dict(arrowstyle='-|>',color=RED,lw=1.1)); axes[1].text(7.6,Y['L12']+1.0,'V from twin\n(labels\nexchanged)',fontsize=5.6,color=RED,va='center')
axes[1].text(4.2,Y['END']-0.9,'read: margin at END',fontsize=5.6,color=GREY,va='center')
# (c) gate steering at END: add +g to the residual before the readers
Y=column(axes[2],'(c) gate steering (+g)',rows,hl='END')
axes[2].annotate('',xy=(3.9,Y['END']),xytext=(6.0,Y['END']),arrowprops=dict(arrowstyle='-|>',color=RED,lw=1.1)); axes[2].text(6.45,Y['END'],'+g, natural dose,\nlayers 20-23',fontsize=5.6,color=RED,va='center',ha='left')
axes[2].text(4.2,Y['END']-1.5,'read: readers\u2019 sink\nand match mass',fontsize=5.6,color=GREY,va='center')
# (d) query-code handoff: patch the two query heads' outputs at END
Y=column(axes[3],'(d) query-code handoff',rows,hl='END')
axes[3].annotate('',xy=(3.9,Y['END']),xytext=(6.0,Y['END']),arrowprops=dict(arrowstyle='-|>',color=RED,lw=1.1)); axes[3].text(6.45,Y['END'],'L22h10, L23h26\noutputs from twin',fontsize=5.6,color=RED,va='center',ha='left')
axes[3].text(4.2,Y['END']-1.5,'read: margin, with\nlater attention held',fontsize=5.6,color=GREY,va='center')
from figcheck import check; check(fig)
fig.savefig('figures/fig18_interventions.pdf',bbox_inches='tight'); fig.savefig('/tmp/fig18.png',dpi=200,bbox_inches='tight'); print('fig18 written')
