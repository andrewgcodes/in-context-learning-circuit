"""One style for every paper figure. import figstyle; figstyle.use()  ->  colours: INTACT, INTERV, CONTROL, PRED, MODEL[name]."""
import matplotlib as mpl, matplotlib.pyplot as plt
INTACT='#111111'; INTERV='#C8102E'; CONTROL='#9A9A9A'; PRED='#1F5FBF'; SECOND='#E08A00'
MODEL={'Qwen3-4B':'#1F5FBF','SmolLM3-3B':'#E08A00','Llama-3.2-3B':'#2E8B57'}
def use():
    mpl.rcParams.update({'font.family':'serif','font.serif':['TeX Gyre Termes','STIXGeneral','DejaVu Serif'],'font.size':9,'axes.titlesize':9.5,'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.7,'xtick.major.width':0.7,'ytick.major.width':0.7,'legend.frameon':False,'figure.dpi':200,'savefig.dpi':300,'savefig.bbox':'tight','pdf.fonttype':42,'axes.grid':False})
def panel_label(ax,s,x=-0.12,y=1.04): ax.text(x,y,s,transform=ax.transAxes,fontweight='bold',fontsize=10,va='bottom',ha='left')
def effect_bar(ax,labels,eff,lo,hi,colors,xlabel='effect on margin (nats)'):
    import numpy as np
    y=np.arange(len(labels))[::-1]
    ax.barh(y,eff,color=colors,height=0.6,xerr=[np.array(eff)-np.array(lo),np.array(hi)-np.array(eff)],error_kw=dict(lw=0.8,capsize=2,ecolor=INTACT))
    ax.axvline(0,color=INTACT,lw=0.7); ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlabel(xlabel)
