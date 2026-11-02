"""Fig 4: the gate. (a) D145 projection on g at L23 (slotgate one=unique / two=recurring) under knocks; (b,c) D66b +g at L23 on unique prompts: per-head change in sink-minus-match and match-minus-other logits (analysis/D66b_gate_logits.json)."""
import os
RUNS=os.environ.get("ICL_RUNS","runs"); DATA=os.environ.get("ICL_DATA","data")
import json,numpy as np,matplotlib.pyplot as plt,figstyle as fs
fs.use()
G=json.load(open(f'{DATA}/D66b_gate_logits.json'))
fig,axs=plt.subplots(1,3,figsize=(7.0,2.4),gridspec_kw={'width_ratios':[1.1,1,1]})
# (a) D145 means (n=30 each)
conds=['none','lookback16','random16','induction16']; names=['intact','duplicate\nheads off','random\nheads off','induction\nheads off']
rec=[18.3,-0.3,17.5,14.0]; uni=[-20.9,-19.1,-21.3,-7.4]
x=np.arange(4); ax=axs[0]
ax.bar(x-0.19,rec,width=0.36,color=fs.INTACT,label='labels repeat'); ax.bar(x+0.19,uni,width=0.36,color=fs.CONTROL,label='labels unique')
ax.axhline(0,color=fs.INTACT,lw=0.6); ax.set_xticks(x); ax.set_xticklabels(names,fontsize=7); ax.set_ylabel('projection on $g$'); ax.legend(fontsize=7,loc='lower left'); ax.set_title('gate signal at the arrow',fontsize=9)
heads=['24:8','28:0','29:11']; hn=['L24h8','L28h0','L29h11']
for ax,idx,title,ylabel in ((axs[1],0,'null logit − match logit','change under +$g$ (nats)'),(axs[2],3,'match logit − other-slot logit','')):
    for j,(c,col,lab) in enumerate((('radd23x1.0_q',fs.INTERV,'+g, natural dose'),('radd23x2.0_q','#F08080','+g, double dose'),('radd23x1.0_qrand',fs.CONTROL,'random, same norm'))):
        v=[G[f'{c}|{h}'] for h in heads]; e=[t[idx] for t in v]
        if idx==0: lo=[t[1] for t in v]; hi=[t[2] for t in v]; ax.bar(np.arange(3)+(j-1)*0.27,e,width=0.25,color=col,label=lab,yerr=[np.array(e)-np.array(lo),np.array(hi)-np.array(e)],error_kw=dict(lw=0.7,capsize=1.5))
        else: ax.bar(np.arange(3)+(j-1)*0.27,e,width=0.25,color=col,label=lab)
    ax.axhline(0,color=fs.INTACT,lw=0.6); ax.set_xticks(range(3)); ax.set_xticklabels(hn,fontsize=7.5); ax.set_title(title,fontsize=9); ax.set_ylabel(ylabel)
axs[1].set_ylim(-5,1); axs[2].set_ylim(-5,1); axs[2].legend(fontsize=6.5,loc='lower left')
for ax,s in zip(axs,'abc'): fs.panel_label(ax,s,x=-0.22)
fig.subplots_adjust(wspace=0.45); fig.savefig('figures/fig4_gate.pdf'); fig.savefig('figures/fig4_gate.png'); print('fig4 written')
