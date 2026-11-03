"""D174 per-head screen: effect of mean-ablating each unnamed (scr_) / named (scrn_) head alone at the query, paired to 'none'.
usage: python analyze_D174.py runs/<run> [runs/<run2> ...] [--thr -0.03] -> ranked table + extended set string."""
import json,sys,collections,numpy as np
dirs=[a for a in sys.argv[1:] if not a.startswith('--')]; thr=float(next((a.split('=')[1] for a in sys.argv if a.startswith('--thr=')),'-0.03'))
rows=[json.loads(l) for d in dirs for l in open(f'{d}/knock.jsonl')]
for r in rows:
    if r.get('ldiff') is None: r['ldiff']=r['tlp']-r['alt_tlp'] if r.get('alt_tlp') is not None else r['tlp']-r['alt']   # margin = log p(target) - log p(best alternative label)
base={}; 
for r in rows:
    if r['cond']=='none': base[(r['seq'],r['kind'])]=r
def fam(k): return 'rare' if 'rare' in k else 'flip' if '_flip_' in k else 'nat' if '_nat_' in k else 'sym'   # D186: nat/flip batteries; the mr/ms columns then read nat/flip
eff=collections.defaultdict(lambda:collections.defaultdict(list)); att=collections.defaultdict(list)
for r in rows:
    c=r['cond']
    if not c.startswith('scr'): continue
    b=base.get((r['seq'],r['kind']))
    if b is None: continue
    eff[c][fam(r['kind'])].append(r['ldiff']-b['ldiff'])
for b in base.values():
    for l,q in b['qatth'].items():
        for h in range(len(q['rep'])): att[(int(l),h)].append((q['rep'][h],q['fresh'][h]))
def ci(x): x=np.array(x); m=x.mean(); se=x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else 0; return m,m-1.96*se,m+1.96*se
tab=[]
for c,d in eff.items():
    l,h=map(int,c.split('_')[1].split(':')); named=c.startswith('scrn')
    F1,F2=('nat','flip') if (d.get('nat') or d.get('flip')) else ('rare','sym'); mr=ci(d[F1])[0] if d[F1] else np.nan; ms=ci(d[F2])[0] if d[F2] else np.nan; both=ci(d[F1]+d[F2])
    a=np.array(att.get((l,h),[(np.nan,np.nan)])).mean(0)
    tab.append((both[0],both[1],both[2],mr,ms,l,h,named,a[0],a[1],len(d[F1])+len(d[F2])))
tab.sort()
print(f'{len(tab)} heads screened; thr {thr}')
print('mean_both  [CI]           rare    sym   head   named  att_rep att_fresh n')
for m,lo,hi,mr,ms,l,h,nm,ar,af,n in (tab if '--all' in sys.argv else tab[:40]): print(f'{m:+.3f} [{lo:+.3f},{hi:+.3f}] {mr:+.3f} {ms:+.3f}  {l:2d}:{h:<2d}  {"N" if nm else "-"}     {ar:.3f}  {af:.3f}  {n}')
un=[t for t in tab if not t[7]]; nm=[t for t in tab if t[7]]
print('\nunnamed: sum of negative effects %+.2f (n=%d, %d below thr); named: sum neg %+.2f (n=%d, %d below thr)'%(sum(t[0] for t in un if t[0]<0),len(un),sum(t[0]<thr for t in un),sum(t[0] for t in nm if t[0]<0),len(nm),sum(t[0]<thr for t in nm)))
print('unnamed heads that help when removed (effect>+0.02):',[(f'{t[5]}:{t[6]}',round(t[0],3)) for t in un if t[0]>0.02][:15])
ext=[f'{t[5]}:{t[6]}' for t in un if t[2]<0 and t[0]<thr]   # CI excludes 0 and effect below thr
print(f'\nEXT ({len(ext)} heads, CI<0 and mean<{thr}):',','.join(ext))
