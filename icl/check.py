"""Post-run invariants for a tagscope/knock output dir. Usage: python3 check.py <dir> [expected_kind,...]
Asserts: every expected kind present; tlp<=tlp1; ntgt>=1 and decoded target non-empty/non-space for icl rows;
equal rows per condition; every knock row has an intact ('none') twin with same seq/kind; prints decoded target per kind."""
import json,sys,collections
d=sys.argv[1]; exp=sys.argv[2].split(',') if len(sys.argv)>2 else []
K=[json.loads(l) for l in open(f'{d}/knock.jsonl')]
kinds=collections.Counter(r['kind'] for r in K); conds=collections.Counter(r['cond'] for r in K)
missing=[k for k in exp if not any(kk==k or kk.startswith(k) for kk in kinds)]
assert not missing, ('expected kinds missing from battery',missing,sorted(kinds))
assert len(set(conds.values()))==1, ('unequal rows per condition',conds)
base={(r['seq'],r['kind']) for r in K if r['cond']=='none'}
assert all((r['seq'],r['kind']) in base for r in K if not r['cond'].startswith('twin_')), 'knock row without intact twin'   # twin_* source rows are scored under the twin's kind
icl=[r for r in K if r['kind'].startswith('icl')]
assert icl, 'no icl rows'
for r in icl:
    assert r['tlp'] is not None and r['ntgt']>=1, ('empty target',r['kind'],r['seq'])
    assert r['tlp']<=r['tlp1']+1e-6, ('seq lp > first-token lp',r['kind'],r['tlp'],r['tlp1'])
    if 'tgt_dec' in r: assert r['tgt_dec'] and r['tgt_dec'].strip(), ('blank decoded target',r['kind'],r['tgt_dec'])
    if r.get('arow') and 'T' in r:   # attention-row loupe: every attended position must lie in the prompt [0,T) (query at T-1)
        for hk,row in r['arow'].items():
            bad=[p for p,_,_ in row['top'] if not (0<=p<r['T'])]
            assert not bad, ('attention-row position outside [0,T)',r['kind'],r['seq'],hk,bad,r['T'])
            assert abs(row['mass']-sum(w for _,_,w in row['top']))<1e-3, ('attention-row mass != sum of top weights',hk)
print('conds',dict(conds)); print('kinds',dict(kinds))
seen=set()
for r in icl:
    if r['cond']=='none' and r['kind'] not in seen:
        seen.add(r['kind']); print(f"  {r['kind']:28s} target={r.get('tgt_dec')!r:14} ids={r.get('tgt_ids')} tlp={r['tlp']} tlp1={r['tlp1']} alt={r.get('alt_tlp')}")
print('CHECK OK',d)
