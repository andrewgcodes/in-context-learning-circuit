#!/usr/bin/env python3
"""Connection-cut arms: paired margin effects per condition vs 'none', by prompt family. usage: analyze_D170.py runs/<run dir> [filter]"""
import json,sys,collections,random
d=sys.argv[1]; flt=sys.argv[2] if len(sys.argv)>2 else ''
rows=[json.loads(l) for l in open(f'{d}/knock.jsonl')]
for r in rows:
    if r.get('ldiff') is None: r['ldiff']=r['tlp']-r['alt_tlp']
def fam(k):
    if 'slotgate' in k: return 'slotgate'
    return 'concept_rare' if 'rare' in k else ('concept_sym' if 'sym' in k else k)
base={(r['seq'],r['kind']):r['ldiff'] for r in rows if r['cond']=='none'}
eff=collections.defaultdict(list)
for r in rows:
    if r['cond']=='none' or flt not in r['cond']: continue
    b=base.get((r['seq'],r['kind']))
    if b is None: continue
    eff[(fam(r['kind']),r['cond'])].append(r['ldiff']-b)
def ci(x,n=2000):
    rng=random.Random(0); m=sorted(sum(rng.choices(x,k=len(x)))/len(x) for _ in range(n)); return m[int(.025*n)],m[int(.975*n)]
basem=collections.defaultdict(list)
for (s,k),v in base.items(): basem[fam(k)].append(v)
for f,v in sorted(basem.items()): print(f'{f:14s} none margin {sum(v)/len(v):+.2f} n={len(v)}')
print()
for (f,c),x in sorted(eff.items()):
    lo,hi=ci(x); print(f'{f:14s} {c:34s} n={len(x):3d} d={sum(x)/len(x):+.3f} [{lo:+.2f},{hi:+.2f}]')
