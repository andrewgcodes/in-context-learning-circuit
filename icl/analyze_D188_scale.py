"""D188 scale census loupe. For each run dir (one model size): intact margins per label kind; R/P split on nat/flip pairs;
single-head knock effects (cond head_L_H) with paired bootstrap CIs on symbol+rare prompts (retrieval) and on R / P (nat/flip pairs).
Readers = heads whose knock LOWERS the margin (CI<0); brakes = heads whose knock RAISES it (CI>0).
usage: python3 analyze_D188_scale.py <dir> [<dir> ...]   (prints one block per dir + a cross-size summary table)"""
import json,sys,gzip,os,collections,re,numpy as np
rng=np.random.default_rng(0)
def ci(x):
    x=np.array(x); b=rng.choice(x,(3000,len(x))).mean(1); return x.mean(),np.percentile(b,2.5),np.percentile(b,97.5)
def load(dd):
    # dd may be 'dirA+dirB' (D188: the sym+nat half and the rare+flip half of one census, identical conds); seqs are keyed (part,seq) so halves never collide,
    # and nat/flip pairing below relies on both halves preserving battery order (rank k of nat in part 0 = rank k of flip in part 1 = the same prompt)
    M=collections.defaultdict(dict); K={}
    for i,d in enumerate(dd.split('+')):
        f=d+'/knock.jsonl'; f,op=(f,open) if os.path.exists(f) else (f+'.gz',gzip.open)
        for l in op(f,'rt'):
            r=json.loads(l)
            if not r['kind'].startswith('icl_concept') or r.get('alt_tlp') is None: continue
            s=(i,r['seq']); M[s][r['cond']]=r['tlp']-r['alt_tlp']; K[s]=r['kind'].replace('icl_concept_','').replace('_12n','')
    return M,K
SUMM=[]
for d in sys.argv[1:]:
    M,K=load(d); name=os.path.basename(d.split('+')[0].rstrip('/'))
    conds=sorted({c for v in M.values() for c in v}); heads=[c for c in conds if c.startswith('head_') or c.startswith('scr_')]
    print('='*100); print(name,'prompts',len(M),'conds',len(conds),'single heads',len(heads))
    bykind=collections.defaultdict(list)
    for s in M: bykind[K[s].split('_')[1]].append(M[s]['none'])
    print('intact margin by label kind:',' '.join('%s %+.2f[%+.2f,%+.2f]'%((k,)+ci(v)) for k,v in sorted(bykind.items())))
    # nat/flip pairs: same concept, same order index
    byc=collections.defaultdict(lambda:collections.defaultdict(list))
    for s in sorted(M):
        c,l=K[s].split('_')[0],K[s].split('_')[1]
        if l in('nat','flip'): byc[c][l].append(M[s])
    pairs=[(c,n,f) for c in byc for n,f in zip(byc[c]['nat'],byc[c]['flip'])]
    R0=[(n['none']+f['none'])/2 for c,n,f in pairs]; P0=[(n['none']-f['none'])/2 for c,n,f in pairs]
    print('nat/flip pairs %d  R (follows demos) %+.2f [%+.2f,%+.2f]  P (follows meaning) %+.2f [%+.2f,%+.2f]  m_flip %+.2f'%((len(pairs),)+ci(R0)+ci(P0)+(np.mean(bykind['flip']),)))
    ret=[s for s in M if K[s].split('_')[1] in('sym','rare')]
    rows=[]
    for h in heads:
        ok=[s for s in ret if h in M[s]]
        if len(ok)<len(ret)*0.9: continue
        dret=ci([M[s][h]-M[s]['none'] for s in ok])
        okp=[(c,n,f) for c,n,f in pairs if h in n and h in f]
        dR=ci([(n[h]+f[h])/2-(n['none']+f['none'])/2 for c,n,f in okp]) if okp else (np.nan,)*3
        dP=ci([(n[h]-f[h])/2-(n['none']-f['none'])/2 for c,n,f in okp]) if okp else (np.nan,)*3
        L,H=(map(int,h[5:].split('_')) if h.startswith('head_') else map(int,h[4:].split(':'))); rows.append((L,H,dret,dR,dP))
    m0=np.mean([M[s]['none'] for s in ret])
    readers=[r for r in rows if r[2][2]<0]; brakes=[r for r in rows if r[2][1]>0]
    print('sym+rare intact %.2f | heads screened %d | readers (knock lowers, CI<0) %d, summed %+.2f, largest %+.2f (%.0f%% of intact) | brakes (knock raises, CI>0) %d, summed %+.2f'%(
        m0,len(rows),len(readers),sum(r[2][0] for r in readers),min((r[2][0] for r in readers),default=0),100*abs(min((r[2][0] for r in readers),default=0))/max(m0,1e-6),len(brakes),sum(r[2][0] for r in brakes)))
    Rreaders=[r for r in rows if not np.isnan(r[3][0]) and r[3][2]<0]; Pbrakes=[r for r in rows if not np.isnan(r[4][0]) and r[4][1]>0]
    print('on word labels: R-readers (dR CI<0) %d summed %+.2f | P-brakes (knock RAISES meaning part, dP CI>0) %d summed %+.2f | P-carriers (dP CI<0) %d summed %+.2f'%(
        len(Rreaders),sum(r[3][0] for r in Rreaders),len(Pbrakes),sum(r[4][0] for r in Pbrakes),len([r for r in rows if not np.isnan(r[4][0]) and r[4][2]<0]),sum(r[4][0] for r in rows if not np.isnan(r[4][0]) and r[4][2]<0)))
    print('top readers (sym+rare):  '+'  '.join('%d:%d %+.2f'%(L,H,d[0]) for L,H,d,_,_ in sorted(rows,key=lambda r:r[2][0])[:12]))
    print('top brakes  (sym+rare):  '+'  '.join('%d:%d %+.2f'%(L,H,d[0]) for L,H,d,_,_ in sorted(rows,key=lambda r:-r[2][0])[:12]))
    print('top P-brakes (dP>0):     '+'  '.join('%d:%d %+.2f'%(L,H,p[0]) for L,H,_,_,p in sorted([r for r in rows if not np.isnan(r[4][0])],key=lambda r:-r[4][0])[:12]))
    print('top R-readers (dR<0):    '+'  '.join('%d:%d %+.2f'%(L,H,q[0]) for L,H,_,q,_ in sorted([r for r in rows if not np.isnan(r[3][0])],key=lambda r:r[3][0])[:12]))
    for c in ['heads_rand','heads_randb','heads_randx']:
        if c in conds: print('control %s (all screened layers\' random heads, same count): dret %+.2f'%(c,np.mean([M[s][c]-M[s]['none'] for s in ret if c in M[s]])))
    lay=collections.defaultdict(float)
    for L,H,d,_,_ in rows: lay[L]+=d[0]
    print('per-layer summed single-head effect (sym+rare): '+' '.join('%d:%+.2f'%(L,v) for L,v in sorted(lay.items())))
    SUMM.append((name,m0,np.mean(bykind['nat']) if 'nat' in bykind else np.nan,np.mean(bykind['flip']) if 'flip' in bykind else np.nan,np.mean(R0) if R0 else np.nan,np.mean(P0) if P0 else np.nan,len(readers),sum(r[2][0] for r in readers),len(brakes),sum(r[2][0] for r in brakes),len(Pbrakes),sum(r[4][0] for r in Pbrakes)))
    json.dump([dict(L=L,H=H,dret=d[0],dret_lo=d[1],dret_hi=d[2],dR=q[0],dP=p[0],dP_lo=p[1],dP_hi=p[2]) for L,H,d,q,p in rows],open(f'analysis/D188_heads_{name}.json','w'))
print('\n'+'='*100+'\nCROSS-SIZE  model | sym+rare | nat | flip | R | P | #readers sum | #brakes sum | #P-brakes sum')
json.dump([dict(zip(['name','symrare','nat','flip','R','P','n_readers','sum_readers','n_brakes','sum_brakes','n_Pbrakes','sum_Pbrakes'],map(lambda x: x if isinstance(x,str) else float(x),r))) for r in SUMM],open('analysis/D188_summary.json','w'),indent=1)
for r in SUMM: print('%-28s %5.2f  %5.2f  %+5.2f  %5.2f  %5.2f | %3d %+6.2f | %3d %+6.2f | %3d %+6.2f'%r)
