"""D188 scale figure: Qwen3 0.6B/1.7B/4B/8B census on concept12n.
usage: python3 fig_scale.py out.pdf name1=label1 name2=label2 ...   (names = analysis/D188_heads_<name>.json + matching row in analysis/D188_summary.json)
Panels: (a) intact margins per label kind vs size; (b) reader/brake population: count and summed single-head effect (sym+rare) vs size;
(c) largest single reader as a fraction of the intact margin (gain bound: more distributed readout at larger size?); (d) brakes on word meaning (sum dP>0) against m_flip."""
import json,sys,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
out=sys.argv[1]; items=[a.split('=') for a in sys.argv[2:]]
S={r['name']:r for r in json.load(open('analysis/D188_summary.json'))}
fig,ax=plt.subplots(1,4,figsize=(15,3.4)); x=np.arange(len(items)); labs=[l for _,l in items]
for k,c in [('symrare','#555'),('nat','#1f77b4'),('flip','#d62728')]:
    ax[0].plot(x,[S[n][k] for n,_ in items],'o-',color=c,label={'symrare':'invented labels (sym+rare)','nat':'word labels, natural','flip':'word labels, flipped'}[k])
ax[0].axhline(0,color='k',lw=.5); ax[0].set_ylabel('intact margin (nats)'); ax[0].legend(fontsize=7,frameon=False); ax[0].set_title('(a) behaviour vs size',fontsize=9)
H={n:json.load(open(f'analysis/D188_heads_{n}.json')) for n,_ in items}
rd=[[h for h in H[n] if h['dret_hi']<0] for n,_ in items]; br=[[h for h in H[n] if h['dret_lo']>0] for n,_ in items]
ax[1].bar(x-0.18,[len(r) for r in rd],0.36,color='#1f77b4',label='readers (knock lowers)'); ax[1].bar(x+0.18,[len(b) for b in br],0.36,color='#d62728',label='brakes (knock raises)')
ax1b=ax[1].twinx(); ax1b.plot(x-0.18,[sum(h['dret'] for h in r) for r in rd],'v',color='navy'); ax1b.plot(x+0.18,[sum(h['dret'] for h in b) for b in br],'^',color='darkred'); ax1b.set_ylabel('summed effect (nats, markers)',fontsize=8)
ax[1].set_ylabel('heads with CI off zero'); ax[1].legend(fontsize=7,frameon=False); ax[1].set_title('(b) reader / brake populations',fontsize=9)
ax[2].plot(x,[100*abs(min(h['dret'] for h in r))/S[n]['symrare'] for r,(n,_) in zip(rd,items)],'o-',color='#1f77b4'); ax[2].set_ylabel('largest single reader, % of intact'); ax[2].set_title('(c) how distributed the readout is',fontsize=9)
pb=[sum(h['dP'] for h in H[n] if not np.isnan(h['dP_lo']) and h['dP_lo']>0) for n,_ in items]
ax[3].bar(x,pb,0.5,color='#d62728',alpha=.7,label='sum dP of P-brakes'); ax3b=ax[3].twinx(); ax3b.plot(x,[S[n]['flip'] for n,_ in items],'ko-',label='m_flip'); ax3b.axhline(0,color='k',lw=.5); ax3b.set_ylabel('flipped-label margin (nats)',fontsize=8)
ax[3].set_ylabel('brakes on word meaning (nats)'); ax[3].set_title('(d) brakes vs flipped-label margin',fontsize=9)
for a in ax: a.set_xticks(x); a.set_xticklabels(labs)
plt.tight_layout(); plt.savefig(out,bbox_inches='tight'); print('saved',out)
