"""All-head reads for D139a/b (capacity law per head across K; fit on 12-demo, predict 1-slot) and D140 (gate -> sink/match logits per head).
Usage: python3 allheads_law.py law pulled/<D139a dir> pulled/<D139b dir>   |   python3 allheads_law.py gate pulled/<D140 dir>"""
import json,sys,gzip,os,collections,numpy as np
from scipy.optimize import minimize
def rows(d,cond=None):
    f=d+'/knock.jsonl'; f,op=(f,open) if os.path.exists(f) else (f+'.gz',gzip.open)
    for l in op(f,'rt'):
        r=json.loads(l)
        if r.get('arow') and (cond is None or r['cond']==cond): yield r
def masses(r,h):
    tgt=str(r['target']).strip(); tops=r['arow'][h]['top']; mm=mo=0.0; nm=no=0
    for s in r['spans']:
        s0=s[0]; lab=next((t for i,t,p in tops if i==s0),None); p=sum(p for i,t,p in tops if i==s0)
        if lab is None: no+=1; continue
        if str(lab).strip()==tgt: mm+=p; nm+=1
        else: mo+=p; no+=1
    return mm,mo,sum(p for i,t,p in tops if i==0),nm,no
def law(dA,dB):
    A=list(rows(dA,'none')); B=list(rows(dB,'none')) if dB else []
    heads=sorted(A[0]['arow'])
    def pred(th,nm,no): L,Ls,lnr=th; return nm*np.exp(L)/(nm*np.exp(L)+no+np.exp(lnr)+np.exp(Ls))
    out=[]
    for h in heads:
        byK=collections.defaultdict(list)
        for r in A:
            mm,mo,sk,nm,no=masses(r,h)
            if nm: byK[r['meta']['K']].append((mm,nm,no,mo,sk))
        if not byK or np.mean([x[0] for v in byK.values() for x in v])<0.03: continue
        def nll(th):
            L,Ls,lnr=th; e=0
            for v in byK.values():
                for mm,nm,no,mo,sk in v:
                    Z=nm*np.exp(L)+no+np.exp(lnr)+np.exp(Ls); e+=(nm*np.exp(L)/Z-mm)**2+(no/Z-mo)**2+(np.exp(Ls)/Z-sk)**2
            return e
        th=minimize(nll,[2.0,2.0,np.log(20)],method='Nelder-Mead').x
        fitK={K:(np.mean([x[0] for x in v]),np.mean([pred(th,x[1],x[2]) for x in v])) for K,v in sorted(byK.items())}
        err=max(abs(a-b) for a,b in fitK.values())
        oos=''
        if B:
            byKb=collections.defaultdict(list)
            for r in B:
                mm,mo,sk,nm,no=masses(r,h)
                if nm: byKb[r['meta']['K']].append((mm,nm,no,mo,sk))
            if byKb:
                oosK={K:(np.mean([x[0] for x in v]),np.mean([pred(th,x[1],x[2]) for x in v])) for K,v in sorted(byKb.items())}
                def nllb(th):
                    L,Ls,lnr=th; e=0
                    for v in byKb.values():
                        for mm,nm,no,mo,sk in v:
                            Z=nm*np.exp(L)+no+np.exp(lnr)+np.exp(Ls); e+=(nm*np.exp(L)/Z-mm)**2+(no/Z-mo)**2+(np.exp(Ls)/Z-sk)**2
                    return e
                thb=minimize(nllb,[2.0,2.0,np.log(20)],method='Nelder-Mead').x; oos+=f' | own fit 1-slot: L={thb[0]:+.2f} Ls={thb[1]:+.2f} (dLs={thb[1]-th[1]:+.2f}, dL={thb[0]-th[0]:+.2f})'
                oos+=' | OOS 1-slot: '+' '.join(f'K{K} {a:.2f}/{b:.2f}' for K,(a,b) in oosK.items())+f' maxerr {max(abs(a-b) for a,b in oosK.values()):.2f}'
        out.append((err,h,th,fitK,oos))
    for err,h,th,fitK,oos in sorted(out):
        print(f"{h:6s} L={th[0]:+.2f} Ls={th[1]:+.2f} nrest={np.exp(th[2]):.0f} fit: "+' '.join(f'K{K} {a:.2f}/{b:.2f}' for K,(a,b) in fitK.items())+f' maxerr {err:.2f}'+oos)
    print(len(out),'heads with slot mass >=0.03;',sum(1 for e,*_ in out if e<=0.1),'within 0.1')
def gate(d):
    R=collections.defaultdict(dict)
    for r in rows(d): R[(r['seq'],r['kind'])][r['cond']]=r
    conds=sorted({c for v in R.values() for c in v}); print('conds',conds)
    heads=sorted(next(iter(R.values()))['none']['arow'])
    rng=np.random.default_rng(0)
    for c in conds:
        if c=='none': continue
        print('==',c)
        res=[]
        for h in heads:
            ds=collections.defaultdict(list)
            for k,v in R.items():
                if c not in v or 'none' not in v: continue
                a=masses(v['none'],h); b=masses(v[c],h)
                if a[3]==0 or b[3]==0: continue
                eps=1e-4
                sm0=np.log(a[2]+eps)-np.log(a[0]/a[3]+eps); sm1=np.log(b[2]+eps)-np.log(b[0]/b[3]+eps)
                mo0=np.log(a[0]/a[3]+eps)-np.log(a[1]/max(a[4],1)+eps); mo1=np.log(b[0]/b[3]+eps)-np.log(b[1]/max(b[4],1)+eps)
                ds[k[1]].append((sm1-sm0,mo1-mo0,a[2],b[2],a[0],b[0]))
            for kind,v in ds.items():
                v=np.array(v)
                if len(v)<8 or v[:,4].mean()<0.02 and v[:,5].mean()<0.02: continue
                bs=rng.choice(v[:,0],(2000,len(v))).mean(1)
                res.append((kind,h,v[:,0].mean(),np.percentile(bs,2.5),np.percentile(bs,97.5),v[:,1].mean(),v[:,2].mean(),v[:,3].mean(),v[:,4].mean(),v[:,5].mean(),len(v)))
        for kind in sorted({x[0] for x in res}):
            print(' kind',kind)
            for x in sorted([x for x in res if x[0]==kind],key=lambda x:x[2]):
                print(f"  {x[1]:6s} d(Lsink-Lmatch) {x[2]:+.2f} [{x[3]:+.2f},{x[4]:+.2f}]  d(Lmatch-Lother) {x[5]:+.2f}  sink {x[6]:.2f}->{x[7]:.2f} match {x[8]:.2f}->{x[9]:.2f} n={x[10]}")
if __name__=='__main__':
    if sys.argv[1]=='law': law(sys.argv[2],sys.argv[3] if len(sys.argv)>3 else None)
    else: gate(sys.argv[2])
