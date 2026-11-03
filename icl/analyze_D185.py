"""D185 transfer read: python3 analyze_D185.py <pulled_dir> -> per label set (sym/rare/nat/flip) and per concept: intact margin, and d(margin) for every condition
(circuit-only battery-mean, circuit-only resample, random set, all heads, band complements) with paired bootstrap CIs. Also P1-P4 verdict lines."""
import json,sys,gzip,os,collections,numpy as np
d=sys.argv[1]; rng=np.random.default_rng(0)
def rows():
    f=d+'/knock.jsonl'; f,op=(f,open) if os.path.exists(f) else (f+'.gz',gzip.open)
    for l in op(f,'rt'):
        r=json.loads(l)
        if r['kind'].startswith('icl_concept') and r.get('alt_tlp') is not None: yield r['seq'],r['kind'],r['cond'],r['tlp']-r['alt_tlp']
M=collections.defaultdict(dict)
for s,k,c,m in rows(): M[(s,k)][c]=m
def ci(x): x=np.array(x); b=rng.choice(x,(4000,len(x))).mean(1); return x.mean(),np.percentile(b,2.5),np.percentile(b,97.5)
def lab(k): p=k.replace('icl_concept_','').replace('_12n','').split('_'); return p[0],p[1]
conds=sorted({c for v in M.values() for c in v}-{'none'})
LS=['sym','rare','nat','flip']; CS=sorted({lab(k)[0] for _,k in M})
print(f'{len(M)} prompts; conds: {conds}')
print('\n== intact margin (target - alt, nats) by label set x concept')
print('%-6s'%''+''.join('%10s'%c for c in CS)+'%10s'%'ALL')
for l in LS:
    row=[]
    for c in CS+['ALL']:
        x=[v['none'] for (s,k),v in M.items() if lab(k)[1]==l and (c=='ALL' or lab(k)[0]==c)]
        row.append('%+.2f'%np.mean(x) if x else '-')
    print('%-6s'%l+''.join('%10s'%r for r in row))
for cond in conds:
    print(f'\n== {cond}: d(margin) vs intact [95% CI]  (ALL column), per-concept means')
    for l in LS:
        allx=[v[cond]-v['none'] for (s,k),v in M.items() if lab(k)[1]==l and cond in v]
        if not allx: continue
        m,lo,hi=ci(allx); per=' '.join('%s:%+.2f'%(c,np.mean([v[cond]-v['none'] for (s,k),v in M.items() if lab(k)==(c,l) and cond in v])) for c in CS)
        acc=np.mean([v[cond]>0 for (s,k),v in M.items() if lab(k)[1]==l and cond in v])
        print(f'  {l:5s} {m:+.2f} [{lo:+.2f},{hi:+.2f}] acc {acc:.2f} | {per}')
