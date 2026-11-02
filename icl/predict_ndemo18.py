"""D162: compare the predicted predictions (analysis/PRED_ndemo18.json, from 12-demo law constants) with the measured matching-slot mass per head per K in the never-fitted NDEMO=18 regime. Usage: python3 predict_ndemo18.py <pulled dir> <family key>"""
import os
RUNS=os.environ.get("ICL_RUNS","runs"); DATA=os.environ.get("ICL_DATA","data")
import json,sys,collections,numpy as np
from allheads_law import rows,masses
d,fam=sys.argv[1],sys.argv[2]
PRED=json.load(open(f'{DATA}/PRED_ndemo18.json'))[fam]
A=list(rows(d,'none')); heads=sorted(A[0]['arow']); Ks=sorted({r['meta']['K'] for r in A}); nd=collections.Counter((r['meta']['K'],r['meta']['ndemo']) for r in A)
print('items per (K,ndemo):',dict(nd))
errs=[];signed=[];perK=collections.defaultdict(list);rowsout=[]
for h in heads:
    if h not in PRED['K2']: continue
    byK=collections.defaultdict(list)
    for r in A:
        mm,mo,sk,nm,no=masses(r,h)
        if nm: byK[r['meta']['K']].append(mm)
    if not byK: continue
    e=[];s=[]
    for K in Ks:
        if K not in byK: continue
        meas=float(np.mean(byK[K])); pr=PRED[f'K{K}'][h]; e.append(abs(meas-pr)); s.append(meas-pr); perK[K].append((meas,pr))
    errs.append(max(e)); signed.append(np.mean(s)); rowsout.append((h,max(e),[(K,round(float(np.mean(byK[K])),3),PRED[f'K{K}'][h]) for K in Ks if K in byK]))
errs=np.array(errs); signed=np.array(signed)
print(f'{fam}: {len(errs)} heads with 12-demo constants; max|meas-pred| over K: median {np.median(errs):.3f}, <=0.1: {np.mean(errs<=0.1):.0%}, <=0.2: {np.mean(errs<=0.2):.0%}; signed mean (meas-pred) {signed.mean():+.3f} (fraction of heads under-predicted {np.mean(signed>0):.0%})')
for K in Ks:
    m=np.array(perK[K]); print(f'  K{K}: measured median {np.median(m[:,0]):.3f} vs predicted median {np.median(m[:,1]):.3f}; corr {np.corrcoef(m[:,0],m[:,1])[0,1]:.2f}')
rowsout.sort(key=lambda x:-x[1]); print('worst 5:',[(h,round(e,2)) for h,e,_ in rowsout[:5]]); print('example best-mass heads:',[(h,v) for h,e,v in sorted(rowsout,key=lambda x:-max(m for _,m,_ in x[2]))[:3]])
json.dump({'heads':[{'head':h,'maxerr':e,'K':v} for h,e,v in rowsout]},open(f'{DATA}/D162_{fam}_pred_vs_meas.json','w'))
