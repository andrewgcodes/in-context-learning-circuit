"""Fig 23: one census at four Qwen3 sizes. (a) intact margins vs size; (b) reader and brake populations (summed single-head effect, count in the bar); (c) brakes on word meaning against the flipped-label margin.
data: $ICL_DATA/D188_summary.json + D188_heads_<name>.json (analyze_D188_scale.py)."""
import json,os,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, figstyle as fs
fs.use(); A=os.environ.get('ICL_DATA','../data')
names=['Qwen3-0.6B-Base_lbD188SC06c','Qwen3-1.7B-Base_lbD188SC17c','Qwen3-4B-Base_lbD188SC4','Qwen3-8B-Base_lbD188SC8b']; labs=['0.6B','1.7B','4B','8B']; size=np.array([0.6,1.7,4,8])
S={r['name']:r for r in json.load(open(f'{A}/D188_summary.json'))}; H={n:json.load(open(f'{A}/D188_heads_{n}.json')) for n in names}
rd=[[h for h in H[n] if h['dret_hi']<0] for n in names]; br=[[h for h in H[n] if h['dret_lo']>0] for n in names]
pb=[[h for h in H[n] if not np.isnan(h['dP_lo']) and h['dP_lo']>0] for n in names]
fig,axs=plt.subplots(1,3,figsize=(7.2,2.5),gridspec_kw={'width_ratios':[1,1,1]})
ax=axs[0]
for k,c,l in [('symrare',fs.INTACT,'invented labels'),('nat',fs.PRED,'word labels, natural'),('flip',fs.INTERV,'word labels, flipped')]:
    ax.plot(size,[S[n][k] for n in names],'o-',color=c,lw=1.2,ms=4,label=l)
ax.axhline(0,color=fs.CONTROL,lw=0.6,ls=':'); ax.set_xscale('log'); ax.set_xticks(size); ax.set_xticklabels(labs); ax.minorticks_off()
ax.set_ylabel('intact margin (nats)'); ax.set_xlabel('parameters'); ax.legend(loc='upper left',handlelength=1.5); ax.set_title('behaviour grows with size'); fs.panel_label(ax,'a')
ax=axs[1]; x=np.arange(4)
sr=[sum(h['dret'] for h in r) for r in rd]; sb=[sum(h['dret'] for h in b) for b in br]
ax.bar(x,sr,0.6,color=fs.PRED,label='readers (removal lowers)'); ax.bar(x,sb,0.6,color=fs.INTERV,label='brakes (removal raises)')
for i in range(4): ax.text(x[i],sr[i]-0.25,str(len(rd[i])),ha='center',va='top',fontsize=7.5,color=fs.PRED); ax.text(x[i],sb[i]+0.15,str(len(br[i])),ha='center',va='bottom',fontsize=7.5,color=fs.INTERV)
ax.axhline(0,color=fs.INTACT,lw=0.7); ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel('summed single-head effect (nats)'); ax.set_ylim(-7.6,7.4)
ax.legend(loc='upper left',handlelength=1.2); ax.set_title('both populations at every size'); fs.panel_label(ax,'b')
ax.set_xlabel('number = heads with interval off zero',fontsize=7,color=fs.CONTROL)
ax=axs[2]
spb=[sum(h['dP'] for h in p) for p in pb]; mf=[S[n]['flip'] for n in names]
ax.bar(x,spb,0.6,color=fs.INTERV,alpha=0.85,label='brakes on word meaning\n(summed $\\Delta P$, left)')
for i in range(4): ax.text(x[i],spb[i]+0.08,str(len(pb[i])),ha='center',va='bottom',fontsize=7.5,color=fs.INTERV)
ax.set_ylabel('hold on word meaning (nats)'); ax.set_ylim(0,7.2); ax.set_xticks(x); ax.set_xticklabels(labs)
ax2=ax.twinx(); ax2.plot(x,mf,'o-',color=fs.INTACT,lw=1.2,ms=4,label='flipped-label margin (right)'); ax2.axhline(0,color=fs.CONTROL,lw=0.6,ls=':'); ax2.set_ylim(-0.25,1.8); ax2.set_ylabel('flipped-label margin (nats)'); ax2.spines['top'].set_visible(False)
h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels(); ax.legend(h1+h2,l1+l2,loc='upper left',handlelength=1.2)
ax.set_title('brakes rise with the flipped margin'); fs.panel_label(ax,'c')
plt.tight_layout(w_pad=1.2); plt.savefig('figures/fig23_scale.pdf'); print('saved figures/fig23_scale.pdf')
