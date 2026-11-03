"""D185 nat/flip decomposition. Same sentences+query under nat and flip labels, so per prompt
  R = (m_nat + m_flip)/2  = the part of the margin that follows the DEMONSTRATED mapping (retrieval),
  P = (m_nat - m_flip)/2  = the part that follows the labels' MEANING regardless of the demos (prior).
For every condition prints dR, dP vs intact with paired bootstrap CIs. usage: python3 analyze_D185_decomp.py <dir>"""
import json,sys,gzip,os,collections,numpy as np
d=sys.argv[1]; rng=np.random.default_rng(0)
M=collections.defaultdict(dict)
f=d+'/knock.jsonl'; f,op=(f,open) if os.path.exists(f) else (f+'.gz',gzip.open)
for l in op(f,'rt'):
    r=json.loads(l)
    if not r['kind'].startswith('icl_concept') or r.get('alt_tlp') is None: continue
    p=r['kind'].replace('icl_concept_','').replace('_12n','').split('_')
    if p[1] in ('nat','flip'): M[(r['seq'],p[0],p[1])][r['cond']]=r['tlp']-r['alt_tlp']
# pair nat/flip prompts: they are built consecutively from the same demos; pair by (concept, order index within concept)
byc=collections.defaultdict(lambda:collections.defaultdict(list))
for (s,c,l),v in sorted(M.items()): byc[c][l].append(v)
def ci(x): x=np.array(x); b=rng.choice(x,(4000,len(x))).mean(1); return x.mean(),np.percentile(b,2.5),np.percentile(b,97.5)
conds=sorted({c for v in M.values() for c in v})
pairs=[(c,n,f_) for c in byc for n,f_ in zip(byc[c]['nat'],byc[c]['flip'])]
print(len(pairs),'nat/flip pairs')
R0=[(n['none']+f_['none'])/2 for c,n,f_ in pairs]; P0=[(n['none']-f_['none'])/2 for c,n,f_ in pairs]
print('intact: R (follows demos) %+.2f [%+.2f,%+.2f]   P (follows meaning) %+.2f [%+.2f,%+.2f]'%(ci(R0)+ci(P0)))
for c_ in sorted(byc): 
    r=[(n['none']+f_['none'])/2 for c,n,f_ in pairs if c==c_]; p=[(n['none']-f_['none'])/2 for c,n,f_ in pairs if c==c_]; print('   %-9s R %+.2f  P %+.2f'%(c_,np.mean(r),np.mean(p)))
print('\ncondition                      dR [CI]                 dP [CI]        | per-concept dR / dP')
for cond in conds:
    if cond=='none': continue
    ok=[(c,n,f_) for c,n,f_ in pairs if cond in n and cond in f_]
    if not ok: continue
    dR=[(n[cond]+f_[cond])/2-(n['none']+f_['none'])/2 for c,n,f_ in ok]; dP=[(n[cond]-f_[cond])/2-(n['none']-f_['none'])/2 for c,n,f_ in ok]
    per=' '.join('%s:%+.2f/%+.2f'%(c_,np.mean([x for (c,_,_),x in zip(ok,dR) if c==c_]),np.mean([x for (c,_,_),x in zip(ok,dP) if c==c_])) for c_ in sorted(byc))
    print('%-28s %+.2f [%+.2f,%+.2f]   %+.2f [%+.2f,%+.2f] | %s'%((cond,)+ci(dR)+ci(dP)+(per,)))
