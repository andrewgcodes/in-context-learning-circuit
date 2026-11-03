"""Recompute, for every run directory, the intact margin and every condition's paired effect with bootstrap intervals.
Usage: python effects.py effects.tsv runs/example [more run dirs...]   (defaults to every directory under runs/)"""
import json,gzip,os,sys,glob,collections,numpy as np
rng=np.random.default_rng(0)
def rows(d):
    f=d+'/knock.jsonl'; op=open
    if not os.path.exists(f): f+='.gz'; op=lambda p: gzip.open(p,'rt')
    if not os.path.exists(f): return None
    out=[]
    with op(f) as fh:
        for l in fh:
            try: out.append(json.loads(l))
            except Exception: pass
    return out
def ci(x):
    x=np.asarray(x); b=rng.choice(x,(4000,len(x))).mean(1); return x.mean(),np.percentile(b,2.5),np.percentile(b,97.5)
W=open(sys.argv[1],'w'); W.write('dir\tkindgroup\tcond\tn\tintact_margin\teffect\tlo\thi\tacc_intact\tacc_cond\tmargin_def\n')
DIRS=sys.argv[2:] or sorted(glob.glob('runs/*'))
for d in DIRS:
    R=rows(d)
    if not R: continue
    icl=[r for r in R if r['kind'].startswith('icl') and r.get('tlp') is not None]
    if not icl: continue
    hasalt=all(r.get('alt_tlp') is not None for r in icl); mdef='tlp-alt_tlp' if hasalt else 'tlp'
    M=lambda r: r['tlp']-(r['alt_tlp'] if hasalt else 0)
    base={(r['seq'],r['kind']):M(r) for r in icl if r['cond']=='none'}
    groups=collections.defaultdict(list)
    for r in icl: groups[(r['kind'].split('_K')[0] if '_K' in r['kind'] else r['kind'])].append(r)
    groups['ALL']=icl
    for g,rs in groups.items():
        conds=collections.defaultdict(list)
        for r in rs:
            k=(r['seq'],r['kind'])
            if k in base and r['cond']!='none': conds[r['cond']].append((base[k],M(r)))
        bm=[v for (s,k),v in base.items() if g=='ALL' or (k.split('_K')[0] if '_K' in k else k)==g]
        if not bm: continue
        im,_,_=ci(bm)
        W.write(f"{os.path.basename(d)}\t{g}\tnone\t{len(bm)}\t{im:+.3f}\t\t\t\t{np.mean(np.array(bm)>0):.2f}\t\t{mdef}\n")
        for c,pairs in conds.items():
            b=np.array([p[0] for p in pairs]); m=np.array([p[1] for p in pairs]); e,lo,hi=ci(m-b)
            W.write(f"{os.path.basename(d)}\t{g}\t{c}\t{len(pairs)}\t{b.mean():+.3f}\t{e:+.3f}\t{lo:+.3f}\t{hi:+.3f}\t{np.mean(b>0):.2f}\t{np.mean(m>0):.2f}\t{mdef}\n")
W.close(); print('written',sys.argv[1])
