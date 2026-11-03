"""D188 gate dose: per label kind (slotgate one = unique labels, two = recurring), margin tlp-alt_tlp under none vs +g (radd_q) vs random direction (qrand) vs +g at demo arrows; paired bootstrap CIs. usage: python3 analyze_D188_gate.py <pulled_dir> [...]"""
import json,sys,gzip,os,collections,numpy as np
rng=np.random.default_rng(0)
def ci(x):
    x=np.array(x); b=[rng.choice(x,len(x)).mean() for _ in range(3000)]; return x.mean(),np.percentile(b,2.5),np.percentile(b,97.5)
for d in sys.argv[1:]:
    f=d+'/knock.jsonl'; op=open
    if not os.path.exists(f): f+='.gz'; op=gzip.open
    M=collections.defaultdict(dict); K={}; G=collections.defaultdict(dict)
    for l in op(f,'rt'):
        r=json.loads(l)
        if r.get('alt_tlp') is None: continue
        M[r['seq']][r['cond']]=r['tlp']-r['alt_tlp']; K[r['seq']]=r['kind']
        if r.get('gproj'): G[r['seq']][r['cond']]=list(r['gproj'].values())[0]
    conds=sorted({c for s in M for c in M[s] if c=='none' or c.startswith('radd')})
    print('='*90); print(os.path.basename(d),'prompts',len(M))
    for kind in sorted(set(K.values())):
        S=[s for s in M if K[s]==kind and all(c in M[s] for c in conds)]
        print(f'  {kind} n={len(S)}')
        base=np.array([M[s]['none'] for s in S]); m,lo,hi=ci(base); print(f'    none margin {m:+.2f} [{lo:+.2f},{hi:+.2f}]  gproj {np.mean([G[s].get("none",np.nan) for s in S]):+.1f}')
        for c in conds:
            if c=='none': continue
            d_=np.array([M[s][c]-M[s]['none'] for s in S]); m,lo,hi=ci(d_)
            print(f'    {c:24s} d margin {m:+.2f} [{lo:+.2f},{hi:+.2f}]  gproj {np.mean([G[s].get(c,np.nan) for s in S]):+.1f}')
