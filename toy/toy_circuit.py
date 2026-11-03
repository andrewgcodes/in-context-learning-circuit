"""Build-from-parts A: a 3-layer attention-only transformer whose weights are SET BY HAND to be the paper's circuit
(L1 writer + duplicate heads, L2 pooling head -> g, L3 readers with a sink at position 0). Residual blocks: C class | V label | P position | T type | D dup flag | G gate.
Runs the paper's measurements on it: read law (eq. law) vs measured mass, unique-vs-recurring gate collapse, dup-knock / +g rescue / g-block from reader queries,
gain bound (margin vs number of readers under a final RMSNorm). Writes toy_results.json + toy_fig.pdf. CPU, seconds."""
import torch,json,math,itertools,random,sys
torch.set_default_dtype(torch.float64); OUT=sys.argv[1] if len(sys.argv)>1 else '.'
K_MAX,VOC,PMAX=8,16,128; T=4  # types: 0 bos 1 sent 2 arrow 3 label
oC,oV,oP,oT,oD,oG=0,K_MAX,K_MAX+VOC,K_MAX+VOC+PMAX,K_MAX+VOC+PMAX+T,K_MAX+VOC+PMAX+T+1; oS=oG+1; oQ=oS+K_MAX; D=oQ+K_MAX  # S slot key (written at labels), Q query code (written at arrows)
def embed(seq):
    x=torch.zeros(len(seq),D)
    for p,(t,c,v) in enumerate(seq):
        x[p,oP+p]=1; x[p,oT+t]=1
        if t==1: x[p,oC+c]=1
        if t==3: x[p,oV+v]=1
    return x
def attn(x,Wq,Wk,Wv,Wo,scale=1.0,bias=None):
    q,k,v=x@Wq,x@Wk,x@Wv; s=scale*q@k.T
    if bias is not None: s=s+bias(x)
    n=x.shape[0]; s=s.masked_fill(torch.triu(torch.ones(n,n,dtype=bool),1),-1e9); a=s.softmax(-1); return a@v@Wo,a
def shift(d): # Wq maps position one-hot p -> one-hot p-d (so label at p attends sentence at p-d)
    W=torch.zeros(D,PMAX)
    for p in range(d,PMAX): W[oP+p,p-d]=1
    return W
Wk_pos=torch.zeros(D,PMAX); Wk_pos[oP:oP+PMAX]=torch.eye(PMAX)
def copyC(dst): Wv=torch.zeros(D,K_MAX); Wv[oC:oC+K_MAX]=torch.eye(K_MAX); Wo=torch.zeros(K_MAX,D); Wo[:,dst:dst+K_MAX]=torch.eye(K_MAX); return Wv,Wo
class Toy:
    def __init__(s,H=1,beta=3.0,Ls0=4.0,alpha=4.0,gamma=1.0,ov=1.0,noise=0.0): s.H,s.beta,s.Ls0,s.alpha,s.gamma,s.ov,s.noise=H,beta,Ls0,alpha,gamma,ov,noise; s.knock=set(); s.gadd=0.0; s.gblock=False
    def forward(s,seq):
        x=embed(seq); n=x.shape[0]; rec={}
        # L1 writer heads: label<-sentence (p-2) and arrow<-sentence (p-1), copy class block (slot key / query code)
        if 'writer' not in s.knock:
            for d in (2,1):
                Wv,Wo=copyC(oS if d==2 else oQ); o,_=attn(x,shift(d),Wk_pos,Wv,Wo,scale=50.0); mask=torch.tensor([t==(3 if d==2 else 2) for t,_,_ in seq],dtype=x.dtype)[:,None]; x=x+o*mask
        # L1 duplicate head: at label p, q=k=label identity among earlier LABEL tokens; bos key gives the no-match fallback; value = 1 at labels -> D flag
        if 'dup' not in s.knock:
            Wq=torch.zeros(D,VOC+1); Wq[oV:oV+VOC,:VOC]=torch.eye(VOC); Wq[oT+3,VOC]=1
            Wk=torch.zeros(D,VOC+1); Wk[oV:oV+VOC,:VOC]=torch.eye(VOC); Wk[oT+0,VOC]=0.5   # bos logit 10 (absorbs no-match) ; matching earlier label logit 20
            Wv=torch.zeros(D,1); Wv[oT+3,0]=1; Wo=torch.zeros(1,D); Wo[0,oD]=1
            bias=lambda x_: ((x_[:,oT+3][:,None]*x_[:,oT+1][None,:]+x_[:,oT+3][:,None]*x_[:,oT+2][None,:])*-1e9)+torch.eye(n)*-1e9  # labels ignore sentences/arrows/self
            o,_=attn(x,Wq,Wk,Wv,Wo,scale=20.0,bias=bias); x=x+o*torch.tensor([t==3 for t,_,_ in seq],dtype=x.dtype)[:,None]
        # L2 pooling head: arrow attends uniformly to label tokens, averages D -> G
        if 'pool' not in s.knock:
            Wq=torch.zeros(D,1); Wq[oT+2,0]=1; Wk=torch.zeros(D,1); Wk[oT+3,0]=1; Wv=torch.zeros(D,1); Wv[oD,0]=s.gamma; Wo=torch.zeros(1,D); Wo[0,oG]=1
            o,_=attn(x,Wq,Wk,Wv,Wo,scale=30.0); x=x+o*torch.tensor([t==2 for t,_,_ in seq],dtype=x.dtype)[:,None]
        x[-1,oG]+=s.gadd; rec['g']=float(x[-1,oG])
        # L3 readers: q = beta*class + G*(gate dir); keys = slot class (labels), sink key at bos: logit Ls0 - alpha*G; values = label identity -> logits block
        logits=torch.zeros(VOC); masses=[]
        for h in range(s.H):
            if ('reader',h) in s.knock: continue
            Wq=torch.zeros(D,K_MAX+1); Wq[oQ:oQ+K_MAX,:K_MAX]=s.beta*torch.eye(K_MAX); Wq[oG,K_MAX]=0.0 if s.gblock else -s.alpha
            Wk=torch.zeros(D,K_MAX+1); Wk[oS:oS+K_MAX,:K_MAX]=torch.eye(K_MAX); Wk[oT+0,K_MAX]=1.0
            Wv=torch.zeros(D,VOC); Wv[oV:oV+VOC]=torch.eye(VOC)*s.ov; Wo=torch.eye(VOC)
            bias=lambda x_: (x_[:,oT+0][None,:]*s.Ls0)+(x_[:,oT+1][None,:]+x_[:,oT+2][None,:])*-1e9*0   # sink base logit; sentences/arrows keep logit 0 (= n_rest)
            o,a=attn(x,Wq,Wk,Wv,Wo,bias=bias); logits=logits+o[-1]; masses.append(a[-1])
        if masses: a=torch.stack(masses).mean(0); rec['m_sink']=float(a[0]); rec['att']=a
        rho=math.sqrt(s.noise**2+float((logits**2).mean())) if s.noise else 1.0   # final RMSNorm with an unrelated residual of RMS 'noise'
        rec['logits']=logits/rho; rec['rho']=rho; rec['m_match']=float(sum(a[p] for p,(t,_,c) in enumerate(seq) if t==3 and seq[p-2][1]==0)) if masses else 0.0; return rec
def prompt(K,n_m,rule='two',seed=0):
    """K classes, n_m demos per class; 'two' = every class reuses one label (recurring); 'one' = every demo gets a fresh label (unique). Query = class 0, target = its label (for 'one': the label of class 0's demo... unique prompts have n_m=1 per class with fresh labels: the target is that demo's label)."""
    r=random.Random(seed); labels=r.sample(range(VOC),K*n_m if rule=='one' else K); seq=[(0,0,0)]; demos=[(c,i) for i in range(n_m) for c in range(K)]; r.shuffle(demos)
    for j,(c,i) in enumerate(demos): lab=labels[c] if rule=='two' else labels[j]; seq+= [(1,c,0),(2,0,0),(3,0,lab)]
    seq+=[(1,0,0),(2,0,0)]; tgt=labels[0] if rule=='two' else labels[[j for j,(c,i) in enumerate(demos) if c==0][0]]
    alts=[l for l in set(labels) if l!=tgt]; return seq,tgt,alts
def margin(rec,tgt,alts): lg=rec['logits']; return float(lg[tgt]-max(lg[a] for a in alts))
R={}
toy=Toy(H=1,beta=4.0,Ls0=5.0,alpha=8.0)
# (1) read law: measured matching mass vs eq. law with the toy's three constants (L=beta, Ls=Ls0-alpha*g, n_rest = #non-slot non-sink positions)
law=[]
for K in (2,3,4,6):
    for n_m in (1,2,3,4):
        seq,tgt,alts=prompt(K,n_m,'two'); rec=toy.forward(seq); att=rec['att']; slots=[p for p,(t,_,_) in enumerate(seq) if t==3]
        match=[p for p in slots if seq[p-2][1]==0]; m=float(att[match].sum()); N=len(slots); nrest=len(seq)-N-1; g=rec['g']
        L=toy.beta; Ls=toy.Ls0-toy.alpha*g; pred=n_m*math.exp(L)/(n_m*math.exp(L)+(N-n_m)+nrest+math.exp(Ls))
        law.append(dict(K=K,n_m=n_m,m=m,pred=pred,g=g,m_sink=rec['m_sink'],margin=margin(rec,tgt,alts)))
R['law']=law; print('law maxerr',max(abs(d['m']-d['pred']) for d in law))
# (2) gate: unique vs recurring, dup knock, +g rescue, g blocked from reader queries
gate={}
for rule in ('one','two'):
    for cond in ('intact','dup_knock','dup_knock+g','dup_knock+g_block','pool_knock','writer_knock'):
        toy.knock=set(); toy.gadd=0.0; toy.gblock=False
        if 'dup' in cond: toy.knock.add('dup')
        if 'pool' in cond: toy.knock.add('pool')
        if 'writer' in cond: toy.knock.add('writer')
        if '+g' in cond: toy.gadd=1.0
        if 'block' in cond: toy.gblock=True
        vals=[];sinks=[]
        for sd in range(20):
            seq,tgt,alts=prompt(6,1,'one',seed=sd) if rule=='one' else prompt(3,2,'two',seed=sd); rec=toy.forward(seq); vals.append(margin(rec,tgt,alts)); sinks.append(rec['m_sink'])
        gate[f'{rule}/{cond}']=dict(margin=sum(vals)/len(vals),m_sink=sum(sinks)/len(sinks))
toy.knock=set(); toy.gadd=0.0; toy.gblock=False; R['gate']=gate
for k,v in gate.items(): print(f"{k:28s} margin {v['margin']:+.2f} sink {v['m_sink']:.2f}")
# (3) gain bound: per-head OV norm capped (ov), unrelated residual of RMS noise -> margin vs number of readers; bound = H * m * ov / rho
gain=[]
for H in (1,2,3,5,8,12,20):
    t=Toy(H=H,beta=4.0,Ls0=5.0,alpha=8.0,ov=0.2,noise=1.0); seq,tgt,alts=prompt(3,2,'two'); rec=t.forward(seq); gain.append(dict(H=H,margin=margin(rec,tgt,alts),bound=H*0.2*rec['m_match']/rec['rho']))
R['gain']=gain; print('gain',[(d['H'],round(d['margin'],2)) for d in gain])
json.dump(R,open(f'{OUT}/toy_results.json','w'),indent=1,default=float)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,3,figsize=(11,3.3))
for K in (2,3,4,6):
    d=[x for x in law if x['K']==K]; ax[0].plot([x['n_m'] for x in d],[x['m'] for x in d],'o-',label=f'K={K} toy'); ax[0].plot([x['n_m'] for x in d],[x['pred'] for x in d],'k--',lw=0.8)
ax[0].set_xlabel('matching slots $n_m$'); ax[0].set_ylabel('mass on matching slots'); ax[0].legend(fontsize=7); ax[0].set_title('read law (dashed = eq. law)')
keys=list(gate); ax[1].barh(range(len(keys)),[gate[k]['margin'] for k in keys]); ax[1].set_yticks(range(len(keys))); ax[1].set_yticklabels(keys,fontsize=6); ax[1].set_xlabel('margin (nats)'); ax[1].set_title('gate interventions')
ax[2].plot([d['H'] for d in gain],[d['margin'] for d in gain],'o-',label='margin'); ax[2].plot([d['H'] for d in gain],[d['bound'] for d in gain],'k--',label='gain bound'); ax[2].set_xlabel('number of readers'); ax[2].legend(fontsize=7); ax[2].set_title('margin vs readers (OV capped)')
plt.tight_layout(); plt.savefig(f'{OUT}/toy_fig.pdf'); print('saved')
