import os
"""Thin driver: load model -> flag direction -> natural scale/means -> battery (tasks.py) -> per-seq head scores -> knockouts -> knock.jsonl -> check.py.
Usage: GRAIL_OUT=dir python3 run.py <model> [families=std,redund,correct,bind,ctl] [smoke] [intact]
Row schema == legacy grail_gpu.py tagscope knock.jsonl, plus 'geom' (key/query geometry) and 'dla' (per-head direct logit attribution)."""
import torch,torch.nn.functional as F,sys,json,random,time,os,subprocess
import numpy as np
from transformers import AutoTokenizer,AutoModelForCausalLM
from tokspans import target_ids
import tasks
from tasks import build_battery,build,LEARN_TASKS,SMOKE_TASKS
from hooks import Hooks,norm_gain,kstyle,attmod,out_proj_matrix,attn_by_layer,down_matrix,qkv_pre,key_dir
from loupes import mlp_neuron_writers,attn_logit_split,value_readout
from score import query_attention_heads,seq_logprob,surprisal,flag_readout,query_attention,dup_head_scores,query_row_top,mlp_direct,slot_rows
name=sys.argv[1]; tag=name.split('/')[-1]; OUT=os.environ.get('GRAIL_OUT') or next((e[4:] for e in sys.argv[3:] if e.startswith('out=')),'') or f'runs/{tag}'; os.makedirs(OUT,exist_ok=True)
ROOT=os.path.dirname(os.path.abspath(__file__))
FAMS=tuple((sys.argv[2] if len(sys.argv)>2 else 'std,redund,correct,bind,ctl').split(',')); EXTRAS=set(sys.argv[3:]); SMOKE='smoke' in EXTRAS
PAIRW_HEADS=[tuple(int(v) for v in t.split(':')) for t in next((e[6:] for e in EXTRAS if e.startswith('pairw=')),'').split(',') if t]   # pairw=22:10,23:26 -> per-row residual write of these heads at the query (loupe)
if any(e=='perm=1' for e in EXTRAS): os.environ['CONCEPT_PERM']='1'
_co=next((e[6:] for e in EXTRAS if e.startswith('order=')),'')
if _co: os.environ['CONCEPT_ORDER']=_co
_bs=next((e[5:] for e in EXTRAS if e.startswith('seed=')),'')
if _bs: os.environ['BATTERY_SEED']=_bs
_nd=next((e[6:] for e in EXTRAS if e.startswith('ndemo=')),'')
if _nd: os.environ['MANYCLASS_NDEMO']=_nd   # D162: never-fitted capacity regime (tasks.fam_manyclass per-class count = ndemo//K)   # seed=1 : fresh prompt draw for held-out confirmation of head sets   # D128: order=lastsame|lastother   # D127: demo-order control (tasks.fam_concept12 re-orders the demonstrations per seq)
dev='cuda' if torch.cuda.is_available() else 'cpu'; tok=AutoTokenizer.from_pretrained(name)
REV=next((e[4:] for e in EXTRAS if e.startswith('rev=')),None)   # rev=step1000 : HF checkpoint revision (training-dynamics studies on Pythia/OLMo)
m=AutoModelForCausalLM.from_pretrained(name,revision=REV,dtype=(torch.float32 if dev=='cpu' else torch.bfloat16),attn_implementation='eager').to(dev).eval()
status=lambda d: open(f'{OUT}/run.status','w').write(json.dumps(d)); t0=time.time()
ATT_TOL=1e-3 if next(m.parameters()).dtype==torch.float32 else 2e-2   # bf16 softmax rows carry ~1e-2 rounding over T~100 tokens
def run(ids):
    with torch.no_grad(): return m(ids[None].to(dev),output_hidden_states=True,output_attentions=True)
H=Hooks(m,late=[]); L=len(H.layers); L0=L//2; late=[l for l in H.SOFT if l>=L0]; H=Hooks(m,late=late); nh=H.nh
print('model',name,'layers',L,'soft',len(H.SOFT),'gdn',len(H.GDN),'late',late,flush=True)
json.dump({'model':name,'L':L,'late':late,'SOFT':H.SOFT,'GDN':H.GDN,'nh':nh,'dev':dev,'dtype':str(next(m.parameters()).dtype),'bos':tok.bos_token_id,'argv':sys.argv},open(f'{OUT}/run_meta.json','w'))
# 1) flag direction U[l]: mean key-input shift at a label position when its demo is an exact repeat of an earlier demo (paired prompts, same length)
T_=json.load(open(tasks._battery('icl_tasks.json'))); T_=LEARN_TASKS(T_)
if SMOKE: T_={k:T_[k] for k in SMOKE_TASKS}
plen=lambda d: len(tok(d[0]+' -> '+d[1]+'\n')['input_ids']); random.seed(1); DEL={l:[] for l in late}
for task,pairs in T_.items():
    for s_ in range(4 if SMOKE else 24):
        d4=random.choice(pairs); same=[d for d in pairs if d!=d4 and plen(d)==plen(d4)]
        if not same: continue
        d1=random.choice(same); d2,d3=random.sample([d for d in pairs if d not in (d1,d4)],2)
        idsS,lpS,_=build(tok,[d1,d2,d3,d4],d4[0]); idsP,lpP,_=build(tok,[d4,d2,d3,d4],d4[0]); lp=lpS[3]; assert lpP[3]==lp
        run(idsS); kS={l:H.kin[l][lp].float().clone() for l in late}; run(idsP)
        for l in late: DEL[l].append(H.kin[l][lp].float()-kS[l])
U={l:F.normalize(torch.stack(DEL[l]).mean(0),dim=0) for l in late}
if 'udir=label' in EXTRAS:   # D152: LABEL-IDENTITY direction instead of the class direction: key-input contrast at a label slot between the two battery labels (' foo' vs ' bar', same length), same prompts otherwise
    DEL={l:[] for l in late}; random.seed(1)
    for task,pairs in T_.items():
        for s_ in range(4 if SMOKE else 24):
            d1,d2,d3,d4=random.sample(pairs,4)
            idsA,lpA,_=build(tok,[d1,d2,d3,(d4[0],'foo')],d4[0]); idsB,lpB,_=build(tok,[d1,d2,d3,(d4[0],'bar')],d4[0]); lp=lpA[3]; assert lpB[3]==lp
            run(idsA); kA={l:H.kin[l][lp].float().clone() for l in late}; run(idsB)
            for l in late: DEL[l].append(kA[l]-H.kin[l][lp].float())
    U={l:F.normalize(torch.stack(DEL[l]).mean(0),dim=0) for l in late}; print('udir=label: U is the foo-vs-bar label-identity direction at label slots',flush=True)
UA={l:F.normalize(torch.stack(DEL[l][0::2]).mean(0),dim=0) for l in late}; UB={l:F.normalize(torch.stack(DEL[l][1::2]).mean(0),dim=0) for l in late}
print('U built',round(time.time()-t0),'nsets',len(DEL[late[0]]),'split-half cos',{l:round(float(UA[l]@UB[l]),3) for l in late},flush=True)
# 2) natural-text scale of the flag readout + head/GDN means for mean-ablation
docs=[]
if os.environ.get('ICL_DOCS'):   # a JSON list of strings, or {'rows':[{'row':{'text':...}}]} as saved from the HF datasets-server
    _d=json.load(open(os.environ['ICL_DOCS'])); docs=[x['row']['text'] for x in _d['rows']] if isinstance(_d,dict) else list(_d)
else:
    from datasets import load_dataset
    docs=[r['text'] for r in load_dataset('NeelNanda/pile-10k',split='train').select(range(300))]
docs=[torch.tensor(tok(d)['input_ids'][:512]) for d in docs if len(d)>1500][:(3 if SMOKE else 16)]
ACC={l:[] for l in late}
def run_acc(ids):
    o=run(ids)
    for l in late: ACC[l].append((H.kin[l][1:].float()@U[l]).cpu())
    return o
H.build_means(run_acc,docs)
KP=next((e[9:] for e in EXTRAS if e.startswith('knockpos=')),'')
if KP=='query': H.KNOCK_POS=[-1]
if 'knockzero' in EXTRAS: H.KNOCK_ZERO=True
print('knock mode pos',H.KNOCK_POS,'zero',H.KNOCK_ZERO,flush=True)
MU={l:torch.cat(ACC[l]).mean().item() for l in late}; SD={l:torch.cat(ACC[l]).std().item() for l in late}
print('natural scale',{l:(round(MU[l],2),round(SD[l],2)) for l in late},flush=True)
# 3) battery
B=build_battery(tok,FAMS,SMOKE)
_kf=next((e[6:] for e in EXTRAS if e.startswith('kinds=')),'')   # D186: kinds=nat_,flip_ keeps only battery rows whose kind contains one of the substrings (screens on a prompt subset)
if _kf: B=[b for b in B if any(k in b.kind for k in _kf.split(','))]; print('kinds filter',_kf,'->',len(B),'rows',flush=True)
_ke=next((e[6:] for e in EXTRAS if e.startswith('every=')),'')   # D186: every=k keeps every k-th row after the kinds filter (cheaper screens)
if _ke: B=B[::int(_ke)]; print('every',_ke,'->',len(B),'rows',flush=True)
# 4) per-head look-back (DS) and induction (IS) scores from attention on every battery sequence, layers below the middle readout layer
RO=late[len(late)//2]; DSm=np.zeros((RO,nh)); ISm=np.zeros((RO,nh)); nrec=0
with open(f'{OUT}/tagscope.jsonl','w') as ft:
    for sid,b in enumerate(B):
        ids=b.ids; T=len(ids); idl=ids.tolist(); dupq=[t for t in range(T) if idl[t] in idl[:t]]
        o=run(ids); lp=F.log_softmax(o.logits[0].float(),-1); TG=flag_readout(H.kin,U,MU,SD,T)
        if dupq:
            AB=attn_by_layer(o,H.SOFT,L)
            for l_,A_ in AB.items():
                if l_>=RO: continue
                lb,ind=dup_head_scores(A_,idl); DSm[l_]+=lb.numpy(); ISm[l_]+=ind.numpy()
            nrec+=1
        ft.write(json.dumps({'seq':sid,'kind':b.kind,'T':T,'dup':dupq,'surp':[round(x,3) for x in surprisal(lp,ids).tolist()],'tag':TG,'spans':[(s0,s1) for (s0,s1,_) in b.spans]})+'\n')
DSm/=max(nrec,1); ISm/=max(nrec,1)
def topk(M,k): idx=np.argsort(-M.ravel())[:k]; return [(int(x//nh),int(x%nh)) for x in idx]
rng=random.Random(0); soft_lo=[l for l in H.SOFT if l<RO]
conds={'lookback8':topk(DSm,8),'lookback16':topk(DSm,16),'induction8':topk(ISm,8),'induction16':topk(ISm,16)}
conds['random8']=[(rng.choice(soft_lo),rng.randrange(nh)) for _ in range(8)]; conds['random16']=[(rng.choice(soft_lo),rng.randrange(nh)) for _ in range(16)]; conds['random8b']=[(rng.choice(soft_lo),rng.randrange(nh)) for _ in range(8)]
conds['both16']=conds['lookback8']+conds['induction8']
if 'intact' in EXTRAS: conds={}
gdn_lo=[l for l in H.GDN if l<RO]
for l in gdn_lo: conds[f'gdn{l}']=[('g',l)]
if gdn_lo: conds['gdn_all']=[('g',l) for l in gdn_lo]
HEADS=next((e[6:] for e in EXTRAS if e.startswith('heads=')),'')   # e.g. "31:30,32:8,33:31" -> condition heads_named (mean-ablate) + two size-matched random draws from the same layers
if HEADS:
    hs=[(int(a),int(b)) for a,b in (x.split(':') for x in HEADS.split(','))]
    for l_,h_ in hs: assert l_ in H.SOFT and 0<=h_<nh, ('bad head',l_,h_)
    conds['heads_named']=hs
    if 'heads_each' in EXTRAS:
        for l_,h_ in hs: conds[f'head_{l_}_{h_}']=[(l_,h_)]
    if 'heads_pairs' in EXTRAS:
        for i_ in range(len(hs)):
            for j_ in range(i_+1,len(hs)): conds[f'pair_{hs[i_][0]}_{hs[i_][1]}__{hs[j_][0]}_{hs[j_][1]}']=[hs[i_],hs[j_]]
    rh=random.Random(13); conds['heads_rand']=[(l_,rh.randrange(nh)) for l_,_ in hs]; conds['heads_randb']=[(l_,rh.randrange(nh)) for l_,_ in hs]
    assert conds['heads_rand']!=hs and conds['heads_randb']!=hs
    allp=[(l_,h_) for l_ in range(len(H.layers)) for h_ in range(nh)]; rx=random.Random(17); pool=[p for p in allp if p not in set(hs)]; conds['heads_randx']=rx.sample(pool,min(len(hs),len(pool)))   # cross-layer, size-matched, no replacement
    print('named heads',hs,flush=True)
DIRK=next((e[4:] for e in EXTRAS if e.startswith('dir=')),'')   # dir=mlp | attn | both : project the flag direction out of late-layer MLP / attention outputs (+ random-direction control)
if DIRK:
    import torch.nn.functional as F_
    def resdir(l):   # residual-space direction whose post-norm image is U[l]
        return F_.normalize(U[l].float().cpu()/norm_gain(H.layers[l]).cpu().clamp_min(1e-6),dim=0)
    tgt_l=[l for l in late if l+1 in U]
    rg=torch.Generator().manual_seed(11); rnd={l:F_.normalize(torch.randn(U[l].numel(),generator=rg),dim=0) for l in tgt_l}
    # dir=mlp | attn | both  -> all late layers; dir=mlp:18-27+29-33+34 -> one condition pair per layer band (hooked layers l, read at l+1)
    kind_,_,bands=DIRK.partition(':')
    def parse_band(b):
        a,_,z=b.partition('-'); a=int(a); z=int(z) if z else a; return [l for l in tgt_l if a<=l<=z]
    BANDS={'':tgt_l} if not bands else {f'_b{b.replace("-","_")}':parse_band(b) for b in bands.split('+')}
    for nm,ls in BANDS.items(): assert ls, ('empty direction band',nm,tgt_l)
    for which in (['mlp','attn'] if kind_=='both' else [kind_]):
        for nm,ls in BANDS.items():
            conds[f'dir_{which}{nm}']=[('d',l,which,'u') for l in ls]; conds[f'dir_{which}{nm}_rand']=[('d',l,which,'r') for l in ls]
    DIRVEC={('u',l):resdir(l+1) for l in tgt_l}; DIRVEC.update({('r',l):rnd[l] for l in tgt_l})
    print('direction conditions',[k for k in conds if k.startswith('dir_')],'layers',tgt_l,flush=True)
KPATH=next((e[6:] for e in EXTRAS if e.startswith('kpath=')),'')   # kpath=31:30,32:8,... (reader heads) -> path-specific flag removal from those heads' KEY inputs (kv groups) / QUERY inputs
if KPATH:
    import torch.nn.functional as F_
    kh=[(int(a_),int(b_)) for a_,b_ in (x.split(':') for x in KPATH.split(','))]; grp_=nh//H.nkv
    for l_,h_ in kh: assert l_ in U and 0<=h_<nh, ('bad kpath head',l_,h_)
    KG={}; KQ={}
    for l_,h_ in kh: KG.setdefault(l_,set()).add(h_//grp_); KQ.setdefault(l_,set()).add(h_)
    KG={l_:sorted(g) for l_,g in KG.items()}; KQ={l_:sorted(g) for l_,g in KQ.items()}
    kg_=torch.Generator().manual_seed(11); KRND={l_:F_.normalize(torch.randn(U[l_].numel(),generator=kg_),dim=0) for l_ in KG}
    ALLG=list(range(H.nkv))
    def kspec(parts):   # parts: [(which, dirkind, groups_by_layer)] -> {layer: {which: (u, groups)}}
        out={}
        for which,dk,G in parts:
            for l_,g in G.items(): out.setdefault(l_,{})[which]=((U[l_] if dk=='u' else KRND[l_]),g)
        return out
    KSPEC={'kpath_k':kspec([('k','u',KG)]),'kpath_k_rand':kspec([('k','r',KG)]),'kpath_k_all':kspec([('k','u',{l_:ALLG for l_ in KG})]),
           'kpath_q':kspec([('q','u',KQ)]),'kpath_v':kspec([('v','u',KG)]),'kpath_v_rand':kspec([('v','r',KG)]),
           'kpath_kv':kspec([('k','u',KG),('v','u',KG)]),'kpath_kqv':kspec([('k','u',KG),('v','u',KG),('q','u',KQ)])}
    for nm in KSPEC: conds[nm]=[('k',nm)]
    print('kpath conditions',{nm:{l_:{w:v[1] for w,v in d.items()} for l_,d in s.items()} for nm,s in KSPEC.items()},flush=True)
PATCHH=next((e[6:] for e in EXTRAS if e.startswith('patch=')),'')   # patch=31:30,32:8,... : activation-patch reader INPUTS from the flag-removed run (source cond psrc=, default dir_mlp_b28_33)
PSRC=next((e[5:] for e in EXTRAS if e.startswith('psrc=')),'dir_mlp_b28_33'); SRC=None
if PATCHH:
    assert PSRC in conds, ('patch source condition missing; add dir=... so that',PSRC,'exists',list(conds))
    assert list(conds).index(PSRC)<len(conds), 'source must run before patch conditions (dict order)'
    ph=[(int(a_),int(b_)) for a_,b_ in (x.split(':') for x in PATCHH.split(','))]; grp_=nh//H.nkv
    for l_,h_ in ph: assert l_ in late and 0<=h_<nh, ('bad patch head',l_,h_)
    PSRC_HEADS=[(l_,h_) for l_,h_ in conds[PSRC]] if conds[PSRC] and isinstance(conds[PSRC][0][0],int) else None   # source is a head-knockout condition
    plo=min(l_ for l_,_ in PSRC_HEADS) if PSRC_HEADS else int(PSRC.split('_b')[-1].split('_')[0])
    PL=[l for l in H.SOFT if l>plo]+([plo] if PSRC_HEADS else [])   # readers downstream of the first removed write (for head sources the same layer's MLP reads it too)
    PL=sorted(set(PL)); PLA=[l for l in PL if l>plo]                 # attention readers strictly after
    prh=random.Random(13); phr=[(l_,prh.randrange(nh)) for l_,_ in ph]; assert phr!=ph
    def heads_spec(hs):
        sp={}
        for l_,h_ in hs:
            d=sp.setdefault(l_,{'k':set(),'q':set(),'v':set()}); d['k'].add(h_//grp_); d['v'].add(h_//grp_); d['q'].add(h_)
        return {l_:{w:sorted(v) for w,v in d.items()} for l_,d in sp.items()}
    PSPEC={'patch_heads':{'attn':heads_spec(ph)},'patch_heads_rand':{'attn':heads_spec(phr)},
           'patch_attn_all':{'attn':{l_:{'k':list(range(H.nkv)),'v':list(range(H.nkv)),'q':list(range(nh))} for l_ in PLA}},
           'patch_mlp_all':{'mlp':PL},'patch_direct':{'direct':True},
           'patch_all':{'attn':{l_:{'k':list(range(H.nkv)),'v':list(range(H.nkv)),'q':list(range(nh))} for l_ in PLA},'mlp':PL,'final':True}}
    if PSRC_HEADS:   # extra readers of interest for a head-knockout source: induction heads, and layer bands
        if 'induction16' in conds: PSPEC['patch_induction16']={'attn':heads_spec(conds['induction16'])}
        if 'patchbands' in EXTRAS:
            nb=4; step=max(1,(len(PLA)+nb-1)//nb)
            for i in range(0,len(PLA),step):
                ls=PLA[i:i+step]; nm=f'L{ls[0]}_{ls[-1]}'
                PSPEC[f'patch_attn_{nm}']={'attn':{l_:{'k':list(range(H.nkv)),'v':list(range(H.nkv)),'q':list(range(nh))} for l_ in ls}}
                PSPEC[f'patch_mlp_{nm}']={'mlp':[l_ for l_ in PL if ls[0]<=l_<=ls[-1]]}
    def build_patch(nm,src):
        sp={}
        for l_,d in PSPEC[nm].get('attn',{}).items():
            for w,idx in d.items(): sp.setdefault(l_,{})[w]=(src['kin'][l_],idx)
        for l_ in PSPEC[nm].get('mlp',[]): sp.setdefault(l_,{})['mlp']=src['min'][l_]
        if PSPEC[nm].get('final'): sp['final']=src['fin']
        if PSPEC[nm].get('direct'): assert src['rem'] is not None, 'source run recorded no removed writes'; sp['final_delta']=-src['rem']
        return sp
    def parse_lrange(part):   # 'mlp:11-19' | 'attn:all' | 'attn:33' -> (kind, [layers]) restricted to the reader layers of that kind
        w,_,rng=part.partition(':'); assert w in ('mlp','attn'), part
        pool=PL if w=='mlp' else PLA
        if rng=='all': ls=list(pool)
        else: a_,_,z_=rng.partition('-'); a_=int(a_); z_=int(z_) if z_ else a_; ls=[l_ for l_ in pool if a_<=l_<=z_]
        assert ls, ('empty layer range',part); return w,ls
    def full_attn(ls): return {l_:{'k':list(range(H.nkv)),'v':list(range(H.nkv)),'q':list(range(nh))} for l_ in ls}
    def spec_for(w,ls): return {'mlp':list(ls)} if w=='mlp' else {'attn':full_attn(ls)}
    def extra_spec(key,default):   # 'key=val' in EXTRAS -> val; bare 'key' -> default; absent -> ''
        return next((e[len(key)+1:] for e in EXTRAS if e.startswith(key+'=')),default if key in EXTRAS else '')
    PLSPEC=extra_spec('patchlayers','mlp:all+attn:all')   # patchlayers=mlp:11-19+attn:20-35 : one condition per layer
    if PLSPEC:
        for part in PLSPEC.split('+'):
            w,ls=parse_lrange(part)
            for l_ in ls: PSPEC[f'patch_{w}_L{l_}']=spec_for(w,[l_])
    PCSPEC=extra_spec('patchcum','')   # patchcum=mlp:11-19+attn:20-35 : cumulative prefix bands L{a}_{k} for k in range (where does the relay accumulate?)
    if PCSPEC:
        for part in PCSPEC.split('+'):
            w,ls=parse_lrange(part)
            for i in range(1,len(ls)): PSPEC[f'patch_{w}_L{ls[0]}_{ls[i]}']=spec_for(w,ls[:i+1])
            for i in range(1,len(ls)-1): PSPEC[f'patch_{w}_L{ls[i]}_{ls[-1]}']=spec_for(w,ls[i:])   # suffix bands too (removing early links)
    def merge_spec(a,b):   # union of two patch specs (attn layer dicts merged by index union; mlp lists unioned)
        out={'mlp':sorted(set(a.get('mlp',[]))|set(b.get('mlp',[]))),'attn':{l_:dict(d) for l_,d in a.get('attn',{}).items()}}
        for l_,d in b.get('attn',{}).items():
            for w,idx in d.items(): out['attn'].setdefault(l_,{})[w]=sorted(set(out['attn'].get(l_,{}).get(w,[]))|set(idx))
        for k_ in ('final','direct'):
            if a.get(k_) or b.get(k_): out[k_]=True
        return {k_:v_ for k_,v_ in out.items() if v_}
    def part_spec(part):   # 'attn:30-35' | 'mlp:30-35' | 'attng:29:6' (kv group + its query heads) | 'attnq:29:6' (query heads only)
        w,_,rest=part.partition(':')
        if part in ('final','direct'): return {part:True}
        if w in ('attng','attnq','attnh'):   # attnh:L:h = one query head
            l_,_,g=rest.partition(':'); l_=int(l_); g=int(g); gq=nh//H.nkv; assert l_ in PLA, part
            if w=='attnh': assert 0<=g<nh, part; return {'attn':{l_:{'q':[g]}}}
            assert 0<=g<H.nkv, part
            d={'q':list(range(g*gq,(g+1)*gq))}
            if w=='attng': d.update(k=[g],v=[g])
            return {'attn':{l_:d}}
        return spec_for(*parse_lrange(part))
    PSSPEC=extra_spec('patchset','')   # patchset=attn:30-35&attng:29:6+attn:29-35&mlp:30-35 : one condition per '+'-group, parts '&'-joined
    if PSSPEC:
        for grp in PSSPEC.split('+'):
            sp={}
            for part in grp.split('&'): sp=merge_spec(sp,part_spec(part))
            PSPEC['patch_'+grp.replace(':','').replace('-','_').replace('&','+')]=sp
    PGSPEC=extra_spec('patchgroups','')   # patchgroups=attn:33+attn:20 : one condition per KV group g (its k,v and the query heads sharing it) at each listed layer
    if PGSPEC:
        gq=nh//H.nkv
        for part in PGSPEC.split('+'):
            w,ls=parse_lrange(part); assert w=='attn', part
            for l_ in ls:
                for g in range(H.nkv): PSPEC[f'patch_attn_L{l_}_g{g}']={'attn':{l_:{'k':[g],'v':[g],'q':list(range(g*gq,(g+1)*gq))}}}
                for g in range(H.nkv): PSPEC[f'patch_attn_L{l_}_g{g}_q']={'attn':{l_:{'q':list(range(g*gq,(g+1)*gq))}}}
        PSPEC['patch_heads_q']={'attn':{l_:{'q':d['q']} for l_,d in heads_spec(ph).items()}}
        PSPEC['patch_heads_kv']={'attn':{l_:{'k':d['k'],'v':d['v']} for l_,d in heads_spec(ph).items()}}
    for nm in PSPEC: conds[nm]=[('p',nm)]
    print('patch conditions',{nm:{k:(v if k!='attn' else {l_:{w:len(i) for w,i in d.items()} for l_,d in v.items()}) for k,v in s.items()} for nm,s in PSPEC.items()},'source',PSRC,'reader layers',PL,flush=True)
NEUR=os.environ.get('GRAIL_NEUR','') or next((e[5:] for e in EXTRAS if e.startswith('neur=')),'')   # e.g. "28:22,31:1,34:20" -> condition neur_top (+ size-matched random neurons at the same layers, 2 draws)
NEURIN=next((e[7:] for e in EXTRAS if e.startswith('neurin=')),'')   # e.g. "neurin=31:1490,32:1154" -> neurons zeroed INSIDE the lookback16 knockout (+ random-neuron control, + neurons alone)
if NEURIN:
    spec=[tuple(int(v) for v in t.split(':')) for t in NEURIN.split(',')]; ff=H.down[spec[0][0]].weight.shape[1] if hasattr(H,'down') and spec[0][0] in H.down else next(iter(H.down.values())).weight.shape[1]
    rr=random.Random(11); rnd=[(l,rr.randrange(ff)) for l,_ in spec]
    conds['neurin_top']=[('m',conds['lookback16'],spec)]; conds['neurin_rand']=[('m',conds['lookback16'],rnd)]; conds['neur_alone']=[('n',l,i) for l,i in spec]
    for l in sorted({l for l,_ in spec}): conds[f'neurin_L{l}']=[('m',conds['lookback16'],[(a,b) for a,b in spec if a==l])]
    print('neuron-in-knockout conditions',{k:v for k,v in conds.items() if k.startswith('neur')},flush=True)
if NEUR:
    spec=[tuple(int(v) for v in t.split(':')) for t in NEUR.split(',')]
    ff=H.down[spec[0][0]].weight.shape[1] if hasattr(H,'down') and spec[0][0] in H.down else next(iter(H.down.values())).weight.shape[1]
    conds['neur_top']=[('n',l,i) for l,i in spec]
    rr=random.Random(7); conds['neur_rand']=[('n',l,rr.randrange(ff)) for l,_ in spec]; conds['neur_randb']=[('n',l,rr.randrange(ff)) for l,_ in spec]
    for l,i in spec: conds[f'neur_L{l}_n{i}']=[('n',l,i)]   # each named neuron alone
    print('neuron conditions',{k:v for k,v in conds.items() if k.startswith('neur')},flush=True)
json.dump({'conds':conds,'DS':DSm.round(4).tolist(),'IS':ISm.round(4).tolist(),'RO':RO},open(f'{OUT}/conds.json','w'))
print('conds',{c:len(v) for c,v in conds.items()},'top lookback',topk(DSm,4),'top induction',topk(ISm,4),flush=True)
# 5) knockouts
Wu=m.get_output_embeddings().weight
FNW=(getattr(getattr(m,'model',m),'norm',None) or getattr(getattr(m,'model',m),'final_layernorm',None) or getattr(getattr(m,'gpt_neox',m),'final_layer_norm',None)).weight.detach().float()   # final RMSNorm/LN gain, folded into the DLA direction (D159)
if 'dlafull=1' in EXTRAS: H.MLPIN_ALL=True   # capture MLP inputs at every layer so the direct-logit accounting is complete
def parse_attrow(spec):   # 'attrow=35:26+35:27+29:all' -> {35:[26,27], 29:'all'}
    out={}
    for part in filter(None,spec.split('+')):
        l_,_,h_=part.partition(':'); l_=int(l_)
        if h_=='all': out[l_]='all'
        elif out.get(l_)!='all': out.setdefault(l_,[]).append(int(h_))
    return out
TWIN=next((e[5:] for e in EXTRAS if e.startswith('twin=')),'')      # twin=swap[,relabel]: counterfactual-source patching from the item's twin prompt (same length, labels swapped / relabelled)
TPOS=next((e[5:] for e in EXTRAS if e.startswith('tpos=')),'query') # tpos=query|labels|all : positions at which the twin's activations are injected
TSET=next((e[8:] for e in EXTRAS if e.startswith('twinset=')),'mlp:11-19+attn:20-35')   # '+'-separated parts: mlp:a-b | attn:a-b | all ; each becomes one condition tp_<twin>_<part>
TWINS=[t for t in TWIN.split(',') if t]
def label_lo(bb,s0):
    """First position of a label slot's window: the token before the label (the arrow), or the arrow before a lone-whitespace token (digit labels tokenize ' ->',' ','7')."""
    lo=max(0,s0-1)
    if lo>0 and tok.decode([int(bb.ids[lo])]).strip()=='': lo-=1
    return lo
GATEFN=''; JBASIS=None; RBASIS=None; PROBE_HEADS=[]   # defaults so non-twin runs do not hit NameError
if TWINS:
    assert TPOS in ('query','labels','all','arrow','labtok','sent','slot1','slot2'), TPOS
    nB=len(B); B=[b for b in B if all(t in b.twins for t in TWINS)]; print('twin battery',len(B),'of',nB,'items have',TWINS,flush=True); assert B, 'no items with the requested twins'
    for tn in TWINS:
        conds[f'twin_{tn}']=[('t',tn)]
        for part in TSET.split('+'): conds[f'tp_{tn}_{part.replace(":","").replace("-","_").replace("@","_").replace("^","_HOLD_")}']=[('tp',tn,part)]   # 'twinpart^selfpart[^selfpart]': hand twinpart from the twin, HOLD selfparts at the item's own intact values (D151: freeze late MLP inputs / attention outputs while the pair's output is handed)
        conds[f'tp_{tn}_exact']=[('tp',tn,'all','all')]   # every reader at every position from the twin: must reproduce the twin's logits at T-1 (first-token check)
    def twin_positions(bb,T,pos=None):
        pos=pos or TPOS
        if pos=='all': return None
        lab=sorted({p for (s0,s1,_) in bb.spans for p in range(label_lo(bb,s0),s1)}); q0=max(s1 for (_,s1,_) in bb.spans)+1
        if pos=='labels': return torch.tensor(lab)
        if pos=='arrow': return torch.tensor(sorted({p for (s0,s1,_) in bb.spans for p in range(label_lo(bb,s0),s0)}))   # the '->' (and a lone-space token) before each label
        if pos=='labtok': return torch.tensor(sorted({p for (s0,s1,_) in bb.spans for p in range(s0,s1)}))   # the label token(s) only
        if pos=='slot1': return torch.tensor(sorted({s0 for (s0,s1,_) in bb.spans}))   # first token of every label slot (D130: is the key on ' z' ...)
        if pos=='slot2': return torch.tensor(sorted({s0+1 for (s0,s1,_) in bb.spans if s1-s0>=2}) or sorted({s0 for (s0,s1,_) in bb.spans}))   # ... or on 'orp'? (falls back to slot1 for single-token labels)
        if pos=='sent': return torch.tensor([p for p in range(q0) if p not in set(lab)])   # demo sentence / newline tokens (everything before the query line that is not a label or '->')
        return torch.arange(q0,T+8)   # query line: after the last demo label through the prompt end (+ shared answer prefix, capped by PATCH_N)
    def shared_prefix(bb,tw):   # answer tokens identical in item and twin (digits ' 7'/' 3' share the leading space) -> patch window extends over them
        a=target_ids(tok,bb.ids.tolist(),bb.target)[0]; c=target_ids(tok,tw.ids.tolist(),tw.target)[0]; n=0
        while n<min(len(a),len(c)) and a[n]==c[n]: n+=1
        return n
    def build_twin_patch(part,src,T):
        part=part.split('@')[0]
        if part=='all': return {**{l_:{'k':list(range(H.nkv)),'v':list(range(H.nkv)),'q':list(range(nh)),'mlp':None} for l_ in set(src['kin'])|set(src['min'])},'final':True}
        w,_,rng=part.partition(':'); a_,_,z_=rng.partition('-'); a_=int(a_) if w not in ('hout','houtu','houtp','houtq','houtqp','houtj','houtjp','houtr','houtrp','hk','hv','hq') else 0; z_=(int(z_) if z_ else a_) if w not in ('hout','houtu','houtp','houtq','houtqp','houtj','houtjp','houtr','houtrp','hk','hv','hq') else 0
        if w=='mlp': return {l_:{'mlp':None} for l_ in src['min'] if a_<=l_<=z_}
        if w=='kv': return {l_:{'k':list(range(H.nkv)),'v':list(range(H.nkv))} for l_ in src['kin'] if a_<=l_<=z_}   # keys/values only (what OTHER positions read from these positions)
        if w in ('hk','hv','hq'):   # hk:28.0,29.2 / hv:... -> KEYS only (hk) or VALUES only (hv) of those KV GROUPS, or hq:29.11,... -> the QUERY input of those HEADS, handed from the twin at the patched positions
            d={}
            for lg in rng.split(','): l_,g_=lg.split('.'); d.setdefault(int(l_),{w[1]:[]})[w[1]].append(int(g_))
            return d
        if w in ('houtr','houtrp'):   # houtr: component inside a RANDOM orthonormal subspace of the same dimension as the probe span (per item, seeded) -> size control for houtj
            d={}
            for lh in rng.split(','): l_,h_=lh.split('.'); d.setdefault(int(l_),{w:[]})[w].append(int(h_))
            return d
        if w in ('houtj','houtjp'):   # houtj:22.10,23.26 -> hand only the component inside the EMPIRICAL probe subspace (span of d(same-slot logit - other-slot logit)/d x_query over probe= heads, intact run, per item; includes qk-norm, RoPE, real keys); houtjp: complement
            d={}
            for lh in rng.split(','): l_,h_=lh.split('.'); d.setdefault(int(l_),{w:[]})[w].append(int(h_))
            return d
        if w in ('houtq','houtqp'):   # houtq:22.10,23.26 -> hand only the component inside the READER-PROBE subspace (span of W_Q^T W_K u over probe= heads); houtqp: its complement
            d={}
            for lh in rng.split(','): l_,h_=lh.split('.'); d.setdefault(int(l_),{w:[]})[w].append(int(h_))
            return d
        if w in ('houtu','houtp'):   # houtu:22.10,23.26 -> hand only the U[l]-component of those heads' residual write from the twin (houtp: only the complement); isolates WHAT a head writes
            d={}
            for lh in rng.split(','): l_,h_=lh.split('.'); d.setdefault(int(l_),{w:[]})[w].append(int(h_))
            return d
        if w=='hout':   # hout:18.30,14.18 -> per-head attention output of those heads handed from the twin at the patched positions
            d={}
            for lh in rng.split(','): l_,h_=lh.split('.'); d.setdefault(int(l_),{'hout':[]})['hout'].append(int(h_))
            return d
        if w=='aout': return {l_:{'aout':None} for l_ in src['aout'] if a_<=l_<=z_}   # attention-sublayer output written into the patched positions (what their heads READ, after o_proj)
        assert w=='attn', part; return {l_:{'k':list(range(H.nkv)),'v':list(range(H.nkv)),'q':list(range(nh))} for l_ in src['kin'] if a_<=l_<=z_}
    def _norm_w(l_):
        n_=getattr(H.layers[l_],'input_layernorm',None) or getattr(H.layers[l_],'ln_1',None); return n_.weight.detach().float() if n_ is not None else torch.ones_like(U[l_])
    _cols=[F.normalize(U[l_].float()*_norm_w(l_).to(U[l_].device),dim=0) for l_ in late if 21<=l_<=29] or [U[late[0]].float()]
    UBASIS=torch.linalg.qr(torch.stack(_cols,1))[0]   # [d, m] orthonormal basis of the class functionals seen by the layer-21..29 readers
    print('houtu class subspace dims',UBASIS.shape[1],'pairwise cos of raw functionals',[round(float(a@b),2) for a,b in zip(_cols[:-1],_cols[1:])],flush=True)
    PROBE_HEADS=[tuple(int(v) for v in t.split(':')) for t in next((e[6:] for e in EXTRAS if e.startswith('probe=')),'').split(',') if t]
    PROBES={}
    if PROBE_HEADS:
        import numpy as _np
        for l_ in late:
            if not (21<=l_<=29): continue
            a_=attmod(H.layers[l_]); Wq=a_.q_proj.weight.detach().float(); Wk=a_.k_proj.weight.detach().float(); gq=getattr(a_,'q_norm',None); gk=getattr(a_,'k_norm',None)
            gq=gq.weight.detach().float() if gq is not None else torch.ones(H.hd,device=Wq.device); gk=gk.weight.detach().float() if gk is not None else torch.ones(H.hd,device=Wq.device)
            xu=F.normalize(U[l_].float(),dim=0).to(Wq.device); wl=_norm_w(l_).to(Wq.device); grp=nh//H.nkv
            for h_ in range(nh):
                g_=h_//grp; ku=gk*(Wk[g_*H.hd:(g_+1)*H.hd]@xu)                        # key of a unit class vector (k_norm gain folded, RoPE ignored)
                p_=(Wq[h_*H.hd:(h_+1)*H.hd].T@(gq*ku))*wl                                # residual-space functional: x·p_ = q_h(x)·k(u) to first order
                PROBES[(l_,h_)]=F.normalize(p_,dim=0).cpu()
        _np.save(f'{OUT}/probes_L21_29.npy',_np.stack([PROBES[k].numpy() for k in sorted(PROBES)])); json.dump([list(k) for k in sorted(PROBES)],open(f'{OUT}/probes_index.json','w'))
        QBASIS=torch.linalg.qr(torch.stack([PROBES[k].to(U[late[0]].device) for k in PROBE_HEADS if k in PROBES],1))[0]
        print('probe subspace from',len(PROBE_HEADS),'reader heads -> dims',QBASIS.shape[1],'cos(probe,class functional) for them',[round(float(PROBES[k]@F.normalize(U[k[0]].float()*_norm_w(k[0]).to(U[k[0]].device),dim=0).cpu()),2) for k in PROBE_HEADS if k in PROBES],flush=True)
    JBASIS=None; RBASIS=None
    GATEFN=next((e[7:] for e in EXTRAS if e.startswith('gatefn=')),'')   # gatefn=<npz>:<layer> (D158): per item, each probe head's GATE functional d[log sink mass - log slot mass]/dx_arrow and MATCH functional d[log same - log other]/dx_arrow at its layer input (intact run); cos with the gate direction g[layer] and the share of g inside each span, vs 20 random same-norm directions -> gatefn.jsonl
    if GATEFN:
        _gp,_gl=GATEFN.rsplit(':',1); GVEC=torch.tensor(np.load(_gp)[f'l{_gl}']).float(); GVEC=GVEC/GVEC.norm(); GLAYER=int(_gl)
    def gate_functional_loupe(ids_in,T,spans,target,sid,kind):
        tgt=str(target).strip(); slots=[s0 for (s0,s1,y) in spans]; same=[s0 for (s0,s1,y) in spans if str(y).strip()==tgt]; other=[s0 for (s0,s1,y) in spans if str(y).strip()!=tgt]
        if GVEC.shape[0]!=m.config.hidden_size: return
        H.clear(); g=GVEC.to(dev); rnd=torch.linalg.qr(torch.randn(g.shape[0],20,generator=torch.Generator().manual_seed(7)))[0].to(dev)
        with torch.enable_grad():
            o=m(ids_in[None].to(dev),output_hidden_states=True,output_attentions=True); hs=o.hidden_states; per={}; GC=[]; MC=[]
            for (l_,h_) in PROBE_HEADS:
                a=o.attentions[l_][0,h_,T-1]; sg=torch.log(a[0]+1e-9)-torch.log(a[slots].sum()+1e-9)
                gg,=torch.autograd.grad(sg,hs[l_],retain_graph=True); gg=F.normalize(gg[0,T-1].float(),dim=0); GC.append(gg)
                rec={'cos_gate':round(float(gg@g),4),'cos_gate_rand':round(float((rnd.T@gg).abs().mean()),4)}
                if same and other:
                    sm=torch.log(a[same].sum()+1e-9)-torch.log(a[other].sum()+1e-9); gm,=torch.autograd.grad(sm,hs[l_],retain_graph=True); gm=F.normalize(gm[0,T-1].float(),dim=0); MC.append(gm)
                    rec.update({'cos_match':round(float(gm@g),4),'cos_match_rand':round(float((rnd.T@gm).abs().mean()),4),'cos_gate_match':round(float(gg@gm),4)})
                per[f'{l_}:{h_}']=rec
        del o
        def share(cols,v):
            if not cols: return None
            B=torch.linalg.qr(torch.stack(cols,1))[0]; return round(float(((B.T@v)**2).sum()),4)
        out={'seq':sid,'kind':kind,'n_heads':len(PROBE_HEADS),'g_in_gatespan':share(GC,g),'g_in_matchspan':share(MC,g),'rand_in_gatespan':round(float(np.mean([share(GC,rnd[:,i]) for i in range(20)])),4) if GC else None,'rand_in_matchspan':round(float(np.mean([share(MC,rnd[:,i]) for i in range(20)])),4) if MC else None,'per_head':per}
        with open(f'{OUT}/gatefn.jsonl','a') as fg: fg.write(json.dumps(out)+'\n')
    def grad_probe_basis(ids_in,T,spans,target):
        """Empirical reader functionals: for each probe head, gradient of [log att-mass on same-class slots - log att-mass on other-class slots] at the query row (T-1)
        w.r.t. that head's layer input at the query position, on the intact forward (eager attention, graph kept). Orthonormal basis of their span [d,m]."""
        H.clear(); tgt=str(target).strip(); same=[s0 for (s0,s1,y) in spans if str(y).strip()==tgt]; other=[s0 for (s0,s1,y) in spans if str(y).strip()!=tgt]
        if not same or not other: return None
        with torch.enable_grad():
            o=m(ids_in[None].to(dev),output_hidden_states=True,output_attentions=True); hs=o.hidden_states; cols=[]
            for (l_,h_) in PROBE_HEADS:
                a=o.attentions[l_][0,h_,T-1]; sc=torch.log(a[same].sum()+1e-9)-torch.log(a[other].sum()+1e-9)
                g,=torch.autograd.grad(sc,hs[l_],retain_graph=True); cols.append(F.normalize(g[0,T-1].float(),dim=0))
        del o; return torch.linalg.qr(torch.stack(cols,1))[0]
    def twin_spec(spec,src):
        sp={}
        for l_,d in spec.items():
            if l_=='final': sp['final']=src['fin']; continue
            for w,idx in d.items():
                if w=='mlp': sp.setdefault(l_,{})['mlp']=src['min'][l_]
                elif w=='aout': sp.setdefault(l_,{})['aout']=src['aout'][l_]
                elif w=='hout': sp.setdefault(l_,{})['hout']=(src['oin'][l_],idx)
                elif w in ('houtu','houtp'): sp.setdefault(l_,{})[w]=(src['oin'][l_],idx,UBASIS)
                elif w in ('houtq','houtqp'): sp.setdefault(l_,{})['houtu' if w=='houtq' else 'houtp']=(src['oin'][l_],idx,QBASIS)
                elif w in ('houtr','houtrp'): assert RBASIS is not None, 'houtr before the intact run'; sp.setdefault(l_,{})['houtu' if w=='houtr' else 'houtp']=(src['oin'][l_],idx,RBASIS)
                elif w in ('houtj','houtjp'): assert JBASIS is not None, 'houtj before the intact run'; sp.setdefault(l_,{})['houtu' if w=='houtj' else 'houtp']=(src['oin'][l_],idx,JBASIS)   # UBASIS: residual-space class subspace (see below)
                else: sp.setdefault(l_,{})[w]=(src['kin'][l_],idx)
        return sp
    print('twin conditions',[k for k in conds if k.startswith('twin_') or k.startswith('tp_')],'tpos',TPOS,flush=True)
SRCT={}; SRCN=None
PPOS=next((e[5:] for e in EXTRAS if e.startswith('ppos=')),'all')   # ppos=query|labels|other|all : restrict knockout-source patch_* conditions to one position class
assert PPOS in ('query','labels','other','all') or PPOS.startswith('slot'), PPOS   # slot<k> = only the k-th demo label (0-based)
def patch_positions(bb,T,pos):
    if pos=='all' or not bb.spans: return None
    lab=sorted({p for (s0,s1,_) in bb.spans for p in range(label_lo(bb,s0),s1)}); q0=max(s1 for (_,s1,_) in bb.spans)+1
    if pos=='labels': return torch.tensor(lab)
    if pos.startswith('slot'):
        k=int(pos[4:]); s0,s1,_=bb.spans[k] if k<len(bb.spans) else bb.spans[-1]
        return torch.tensor(list(range(label_lo(bb,s0),s1)))
    if pos=='query': return torch.arange(q0,T+8)
    return torch.tensor([p for p in range(q0) if p not in set(lab)])   # 'other': sentence / '->' / newline tokens before the query line
if PPOS!='all': print('knockout-source patch positions restricted to',PPOS,flush=True)
AROWH=parse_attrow(next((e[7:] for e in EXTRAS if e.startswith('attrow=')),''))
AROWK=int(next((e[6:] for e in EXTRAS if e.startswith('arowk=')),'8'))
_sr=next((e[5:] for e in EXTRAS if e.startswith('srow=')),'')
SROWL=list(range(int(_sr.split('-')[0]),int(_sr.split('-')[1])+1)) if _sr else []   # srow=10-23 : slot-position attention rows for every head in these layers   # arowk=12 : top-k positions kept per recorded attention row
if AROWH: print('attention-row loupe heads',AROWH,flush=True)
_rd=next((e[6:] for e in EXTRAS if e.startswith('rdump=')),'')
RDUMP=list(range(int(_rd.split('-')[0]),int(_rd.split('-')[1])+1)) if _rd else []
_hd=next((e[6:] for e in EXTRAS if e.startswith('hdump=')),'')
HDUMP=list(range(int(_hd.split('-')[0]),int(_hd.split('-')[1])+1)) if _hd else []   # hdump=8-23 : also save every head's attention output (o_proj input slice) at those positions, for the one-pass class-direction attribution   # rdump=10-27 : save the label-token (and arrow) residual-stream inputs of these layers under cond 'none' to resid_<sid>.npz
_das=[e[5:] for e in EXTRAS if e.startswith('dadd=')]   # dadd=<npz path>:<layer>:<scale> (repeatable): add scale*dir[layer] to the label tokens of the OTHER class, pushed toward the query's class (cond dadd{L}x{scale}), away from it (_away), a random vector of the same norm (_rand), or to the SAME-class tokens (_same)
DADDS={}
for _da in _das:
    _p,_l,_s=_da.rsplit(':',2); _nm=f'dadd{_l}x{_s}'
    if '+' in _l:   # D77 split dose: dadd=<npz>:13+16+19+22:<scale> adds scale*dir[L] at EVERY listed layer (same positions); _l stays the '+'-joined string in the cond name
        _Ls=[int(x) for x in _l.split('+')]; _zz=np.load(_p); DADDS[_nm]=([torch.tensor(_zz[f'l{L}']) for L in _Ls],_Ls,float(_s))
    else:
        _zz=np.load(_p); _pk=[k for k in _zz.files if k.startswith(f'l{_l}_')]   # pool-keyed directions (D123 fit) -> dict pool->vec
        DADDS[_nm]=(({k.split('_',1)[1]:torch.tensor(_zz[k]) for k in _pk} if _pk else torch.tensor(_zz[f'l{_l}'])),int(_l),float(_s))
    conds[_nm]=[('da',_nm,'other')]; conds[_nm+'_away']=[('da',_nm,'away')]; conds[_nm+'_rand']=[('da',_nm,'rand')]; conds[_nm+'_same']=[('da',_nm,'same')]; conds[_nm+'_qflip']=[('da',_nm,'qflip')]; conds[_nm+'_qrand']=[('da',_nm,'qrand')]
    for _m in ('sameq','otherq','qonly','sameqrand','qonlyrand'): conds[_nm+'_'+_m]=[('da',_nm,_m)]   # D70 built-from-parts: key written into same-class slots AND the query pushed toward that class (sameq); wrong-class slots + query (otherq); query only (qonly); random same-norm (…rand)   # qflip: push the QUERY's final position toward the other class; qrand: random same-norm vector there
if DADDS: print('direction add',list(DADDS),flush=True)
RKCOND=next((e[7:] for e in EXTRAS if e.startswith('rkcond=')),'')   # rkcond=lookback16 -> the 'rk' (push+knock) conditions knock that stored condition's heads instead of heads=
_ras=[e[5:] for e in EXTRAS if e.startswith('radd=')]   # radd=<npz>:<layer>:<scale> (D42, functask): add scale*gap[L]*ruledir[L] (anto-minus-copy, unit) pushed toward the OTHER rule (copy items +, anto items -) at the query arrow (_q) or at every demo arrow (_arrows); _qrand/_arrowsrand = same-norm random vector
RADDS={}
for _ra in _ras:
    _p,_l,_s=_ra.rsplit(':',2); _nm=f'radd{_l}x{_s}'; _z=np.load(_p); RADDS[_nm]=(torch.tensor(_z[f'l{_l}']).float()*float(_z[f'gap{_l}']),int(_l),float(_s))
    for _m in ('q','qrand','arrows','arrowsrand'): conds[_nm+'_'+_m]=[('ra',_nm,_m)]
    if (HEADS or RKCOND) and 'raknock' in EXTRAS: conds[_nm+'_q+knock']=[('rk',_nm,'q')]; conds[_nm+'_qrand+knock']=[('rk',_nm,'qrand')]
    if RKCOND:   # D170 rescue-and-block: knock the rkcond heads (duplicate-token heads) and push +g at the query (qplus = forced + sign); block = also remove g from the named readers' QUERY inputs (qplusblock) or a random direction (qplusblockrand)
        for _m in ('qplus','qplusblock','qplusblockrand'): conds[_nm+'_'+_m+'+knock']=[('rk',_nm,_m)]
        for _b in [b_ for b_ in next((e[7:] for e in EXTRAS if e.startswith('gblock=')),'').split(',') if b_]:   # D171 g-route sweep: gblock=23-26,27-30 -> remove g at the QUERY only from ALL heads' queries / all MLP inputs / both in that band (dup heads knocked, +g pushed)
            for _w in ('all','mlp','both'): conds[f'{_nm}_qplusblock{_w}{_b}+knock']=[('rk',_nm,f'qplusblock{_w}{_b}')]
        conds[_nm+'_qplus']=[('ra',_nm,'qplus')]; conds[_nm+'_qplusblock']=[('ra',_nm,'qplusblock')]   # D144: gate push WITH the named heads mean-ablated (does +g still open emission without the released witnesses?)
if RADDS: print('rule-direction add',list(RADDS),flush=True)
TRACE_N=int(next((e[6:] for e in EXTRAS if e.startswith('trace=')),'0'))   # trace=<n>: write per-token/per-layer/per-head shards for the first n prompts (intact + circuit-only)
GPROJ=next((e[6:] for e in EXTRAS if e.startswith('gproj=')),'')   # gproj=<npz>:20,23 -> record the query-arrow residual's projection on the gate direction g at those layers under EVERY condition (D145: does the look-back knock lower g?)
GP={}
if GPROJ:
    _p,_ls=GPROJ.rsplit(':',1); _z=np.load(_p)
    for _l in _ls.split(','): GP[int(_l)]=torch.tensor(_z[f'l{_l}']).float()
    print('gate projection layers',sorted(GP),flush=True)
GV=(GP[23] if 23 in GP else (next(iter(GP.values())) if GP else None))   # gate direction used by the trace shards
if GV is not None: GV=GV/GV.norm()
SINKV=next((e[6:] for e in EXTRAS if e.startswith('sinkv=')),'')   # D170 sink test: sinkv=21-29 -> zero / swap(with position 1) the VALUE at position 0 in those layers; controls: zero at position 1, zero at 0 in all layers
if SINKV:
    _a,_,_z=SINKV.partition('-'); _ls=list(range(int(_a),int(_z or _a)+1))
    conds[f'sinkv_zero_{SINKV}']=[('sv',_ls,0,'zero')]; conds[f'sinkv_swap_{SINKV}']=[('sv',_ls,0,'swap')]; conds[f'sinkv_zero_pos1_{SINKV}']=[('sv',_ls,1,'zero')]
    conds['sinkv_zero_all']=[('sv',list(range(len(H.layers))),0,'zero')]; conds['sinkv_zero_pos1_all']=[('sv',list(range(len(H.layers))),1,'zero')]
    conds[f'sinkv_zeroother_{SINKV}']=[('sv',_ls,0,'zeroother')]; conds['sinkv_zeroother_all']=[('sv',list(range(len(H.layers))),0,'zeroother')]   # D171: zero the value other positions read from position 0, position 0's own output restored
HGAIN=next((e[6:] for e in EXTRAS if e.startswith('hgain=')),'')   # D170 gain test: hgain=24.21,29.11,22.20:2,4,8 -> scale those heads' writes at the query by each gain; hgainrest=<heads> are knocked in the *_restknock variants
if HGAIN:
    _hh,_gg=HGAIN.rsplit(':',1); _gh=[tuple(int(v) for v in t.split('.')) for t in _hh.split(',')]; _gs=[float(g) for g in _gg.split(',')]
    _rest=[tuple(int(v) for v in t.split('.')) for t in next((e[10:] for e in EXTRAS if e.startswith('hgainrest=')),'').split(',') if t]
    _rg=random.Random(23); _rnd=[(l_,_rg.randrange(nh)) for l_,_ in _gh]
    conds['hgain_restknock']=[('hg',_gh,1.0,_rest)]; conds['hgain_setknock']=[('hg',_gh,0.0,[])]
    for g_ in _gs:
        conds[f'hgain_x{g_:g}']=[('hg',_gh,g_,[])]; conds[f'hgain_x{g_:g}_restknock']=[('hg',_gh,g_,_rest)]; conds[f'hgain_x{g_:g}_rand']=[('hg',_rnd,g_,[])]
    print('gain heads',_gh,'rest',len(_rest),flush=True)
CIRC=next((e[5:] for e in EXTRAS if e.startswith('circ=')),'')   # D171 circuit-only: circ=<l:h,...> the named circuit; every OTHER head is mean-ablated at the knock position (circ_only), or the named set is (circ_knock); random same-layer set controls; circmlp=30-35 = MLP layers kept when the rest of the MLPs are also ablated
if CIRC:
    _ch=[(int(a),int(b)) for a,b in (x.split(':') for x in CIRC.split(','))]
    if 'circdup' in EXTRAS: _ch=list(dict.fromkeys(_ch+[tuple(map(int,p)) for p in conds['lookback16']+conds['induction16']]))   # D180: add this run's own duplicate-token heads (lookback16 + induction16) to the named set
    _cs=set(_ch)
    for l_,h_ in _ch: assert l_ in H.SOFT and 0<=h_<nh, ('bad circ head',l_,h_)
    _allp=[(l_,h_) for l_ in range(len(H.layers)) for h_ in range(nh) if l_ in H.SOFT]; _rest=[p for p in _allp if p not in _cs]
    _rc=random.Random(29); _rnd=list({(l_,_rc.randrange(nh)) for l_,_ in _ch}); _rs=set(_rnd); _rrest=[p for p in _allp if p not in _rs]
    _cm=next((e[8:] for e in EXTRAS if e.startswith('circmlp=')),''); _a,_,_z=_cm.partition('-'); _keepm=set(range(int(_a),int(_z or _a)+1)) if _cm else set(); _mrest=[l_ for l_ in range(len(H.layers)) if l_ not in _keepm]
    conds['circ_only']=[('cm',_rest,[])]; conds['circ_knock']=[('cm',_ch,[])]; conds['circ_rand_only']=[('cm',_rrest,[])]; conds['circ_rand_knock']=[('cm',_rnd,[])]
    conds['all_heads_knock']=[('cm',_allp,[])]; conds['circ_only_mlpknock']=[('cm',_rest,_mrest)]; conds['mlp_rest_knock']=[('cm',[],_mrest)]; conds['circ_knock_mlpknock']=[('cm',_ch,_mrest)]
    print('circuit heads',len(_ch),'rest',len(_rest),'random',len(_rnd),'mlp rest',len(_mrest),flush=True)
    _cb=next((e[10:] for e in EXTRAS if e.startswith('circbands=')),'')   # D173: graded complement, e.g. circbands=0-9,10-19,20-27,28-35 -> mean-ablate only the UNNAMED heads inside each layer band
    for _bd in ([x for x in _cb.split(',') if x] if _cb else []):
        _a,_,_z=_bd.partition('-'); _a,_z=int(_a),int(_z or _a); conds[f'rest_band_{_a}-{_z}']=[('cm',[p for p in _rest if _a<=p[0]<=_z],[])]
    _sc=next((e[11:] for e in EXTRAS if e.startswith('circscreen=')),'')   # D174: per-head screen: every UNNAMED head in layers a-b mean-ablated alone at the query (battery means) -> which unnamed heads carry the rest
    if _sc:
        _a,_,_z=_sc.partition('-'); _a,_z=int(_a),int(_z or _a)
        for p in _rest:
            if _a<=p[0]<=_z: conds[f'scr_{p[0]}:{p[1]}']=[('cm',[p],[])]
        for p in _ch:
            if _a<=p[0]<=_z: conds[f'scrn_{p[0]}:{p[1]}']=[('cm',[p],[])]   # named heads screened the same way, for a matched comparison
    if 'circrs' in EXTRAS:   # D173: resample ablation = replace the ablated heads' outputs by the SAME position's outputs from another battery prompt (never off-distribution)
        for _nm in ('circ_only','circ_rand_only','all_heads_knock','circ_only_mlpknock'): conds[_nm+'_rs']=conds[_nm]+[('rs',)]
QOUT=None
if CIRC and ('circmean=battery' in EXTRAS or 'circmean=kind' in EXTRAS or 'circrs' in EXTRAS):   # D173: reference = per-head / per-MLP output at the QUERY position averaged over the battery itself (IOI-style matched distribution), not natural text
    H.clear(); QOUT=[]
    for b_ in B:
        run(b_.ids); _p=len(b_.ids)-1
        QOUT.append({'o':{l:H.oin[l][_p].clone() for l in H.oin},'m':{l:H.MOUT[l][_p].clone() for l in H.MOUT}})
    MEANO_NAT,MEANM_NAT=H.MEANO,H.MEANM
    MEANO_BAT={l:torch.stack([q['o'][l] for q in QOUT]).mean(0) for l in QOUT[0]['o']}; MEANM_BAT={l:torch.stack([q['m'][l] for q in QOUT]).mean(0) for l in QOUT[0]['m']}
    KIND_OF=[b_.kind for b_ in B]   # D186: per-kind references = prompts sharing the same label pair AND label set (sym/rare/nat/flip), so the reference carries the right label words and mapping
    MEANO_KIND={k:{l:torch.stack([q['o'][l] for q,kk in zip(QOUT,KIND_OF) if kk==k]).mean(0) for l in QOUT[0]['o']} for k in set(KIND_OF)}
    MEANM_KIND={k:{l:torch.stack([q['m'][l] for q,kk in zip(QOUT,KIND_OF) if kk==k]).mean(0) for l in QOUT[0]['m']} for k in set(KIND_OF)}
    RS_NEXT=[next(j for j in list(range(i+1,len(B)))+list(range(0,i)) if KIND_OF[j]==KIND_OF[i]) if 'circmean=kind' in EXTRAS else (i+1)%len(B) for i in range(len(B))]   # rs partner: next prompt of the same kind (circmean=kind) else next prompt (D173 behaviour: for interleaved label sets that is the SAME prompt under another label set)
    assert all(RS_NEXT[i]!=i for i in range(len(B))), 'a kind with a single prompt cannot be resampled'
    print('circmean=kind: %d kinds, %d-%d prompts each'%(len(set(KIND_OF)),min(KIND_OF.count(k) for k in set(KIND_OF)),max(KIND_OF.count(k) for k in set(KIND_OF))),flush=True)
    if 'circmean=battery' in EXTRAS: H.MEANO,H.MEANM=MEANO_BAT,MEANM_BAT
    print('battery means over',len(QOUT),'prompts; head-mean shift vs natural (L24 norm ratio)',round(float((MEANO_BAT[24]-MEANO_NAT[24]).norm()/MEANO_NAT[24].norm()),3) if 24 in MEANO_BAT and 24 in MEANO_NAT else None,flush=True)
NMIS=0
with open(f'{OUT}/knock.jsonl','w') as fk:
    for sid,b in enumerate(B):
        if KP=='query': H.KNOCK_POS=[len(b.ids)-1]   # query arrow = last PROMPT token (scoring appends the target)
        elif KP=='slot1': H.KNOCK_POS=[s0 for (s0,s1,_) in b.spans]   # first token of every label slot
        elif KP=='slot2': H.KNOCK_POS=[s0+1 for (s0,s1,_) in b.spans if s1-s0>=2] or [len(b.ids)-1]   # second token of multi-token labels (' z'+'orp'); falls back to the query arrow when no label has two tokens
        b0=b
        for cname,hl in [('none',[])]+list(conds.items()):
            b=b0.twins[hl[0][1]] if (hl and hl[0][0]=='t') else b0     # twin source rows run the twin prompt; everything else the item itself
            ids=b.ids; T=len(ids); idl=ids.tolist(); dup=[t for t in range(T) if idl[t] in idl[:t]]; icl=b.kind.startswith('icl')
            if hl and hl[0][0]=='d': H.clear(); H.set_dirs({l:{which:DIRVEC[(kind_,l)]} for _,l,which,kind_ in hl})
            elif hl and hl[0][0]=='n': H.clear(); H.set_neurons([(l,i) for _,l,i in hl])
            elif hl and hl[0][0]=='m': H.clear(); H.set_knock(hl[0][1]); H.set_neurons(hl[0][2])   # mixed: head knockout + neuron zeroing
            elif hl and hl[0][0]=='k': H.clear(); H.set_kdirs(KSPEC[hl[0][1]])
            elif hl and hl[0][0]=='p':
                assert SRC is not None, 'patch before source'; H.clear(); H.set_patch(build_patch(hl[0][1],SRC)); H.PATCH_POS=patch_positions(b,T,PPOS)
            elif hl and hl[0][0]=='t': H.clear()
            elif hl and hl[0][0]=='da':
                H.clear(); DADD=DADDS[hl[0][1]]; _mode=hl[0][2]; _same=[s0 for (s0,s1,_) in b.spans if tok.decode(idl[s0:s1]).strip()==str(b.target).strip()]; _oth=[s0 for (s0,s1,_) in b.spans if s0 not in _same]
                _lines=[ln for ln in tok.decode(idl).split('\n') if '->' in ln]; _k=[i for i,(s0,_,_) in enumerate(b.spans) if s0 in _same][0]
                try: _qpol=dict((x.strip(),y) for x,y in tasks.SENTPOOL['pool']).get(_lines[_k].split('->')[0].strip(),'P')   # the query's class = class of a same-label demo's sentence
                except Exception: _qpol=b.meta.get('qpol','P')   # non-sentiment families (slotgate pools): no sentiment class; sign defaults to + (direction may not be meaningful there)
                _sgn=(1.0 if _qpol=='P' else -1.0)*(-1.0 if _mode=='away' else 1.0)
                if isinstance(DADD[1],list):   # D77 split dose: the same push at every listed layer (built-from-parts modes only)
                    _ps={'same':_same,'sameq':_same,'sameqrand':_same,'otherq':_oth,'other':_oth,'rand':_oth,'away':_oth}.get(_mode,[])+([len(idl)-1] if _mode in ('sameq','sameqrand','otherq','qonly','qonlyrand','qflip','qrand') else [])
                    _sg=(-_sgn if _mode in ('qflip','qrand') else _sgn)
                    H.DADD={}
                    for _vL,_L in zip(DADD[0],DADD[1]):
                        _v=_vL
                        if _mode.endswith('rand'): _g=torch.Generator().manual_seed(4000+sid+_L); _v=torch.randn(_v.shape,generator=_g); _v=_v/_v.norm()*_vL.norm()
                        H.DADD[_L]=(_v,torch.tensor(_ps,dtype=torch.long),_sg*DADD[2])
                else:
                    _v=DADD[0]
                    if isinstance(_v,dict):   # per-item vector u_A-u_B; the query is drawn from pool A so sign=+1
                        _v=_v[b.meta['A']]-_v[b.meta['B']]; _sgn=(-1.0 if _mode=='away' else 1.0)
                    _vn=_v.norm()
                    if _mode=='rand': _g=torch.Generator().manual_seed(1000+sid); _v=torch.randn(_v.shape,generator=_g); _v=_v/_v.norm()*_vn
                    if _mode in ('qflip','qrand'):
                        if _mode=='qrand': _g=torch.Generator().manual_seed(2000+sid); _v=torch.randn(_v.shape,generator=_g); _v=_v/_v.norm()*_vn
                        H.DADD={DADD[1]:(_v,torch.tensor([len(idl)-1]),-_sgn*DADD[2])}
                    elif _mode in ('sameq','otherq','qonly','sameqrand','qonlyrand'):
                        if _mode.endswith('rand'): _g=torch.Generator().manual_seed(4000+sid); _v=torch.randn(_v.shape,generator=_g); _v=_v/_v.norm()*_vn
                        _ps={'sameq':_same,'sameqrand':_same,'otherq':_oth}.get(_mode,[])+[len(idl)-1]
                        H.DADD={DADD[1]:(_v,torch.tensor(_ps),_sgn*DADD[2])}
                    else: H.DADD={DADD[1]:(_v,torch.tensor(_same if _mode=='same' else _oth),_sgn*DADD[2])}
            elif hl and hl[0][0] in ('ra','rk'):
                H.clear(); RA=RADDS[hl[0][1]]; _mode=hl[0][2]; _v=RA[0]; _sgn=1.0 if b.kind.endswith(('copy','_one')) else -1.0   # push toward the OTHER rule: copy->rule, one-slot->two-slot (gate)
                if 'rand' in _mode and 'block' not in _mode: _g=torch.Generator().manual_seed(3000+sid); _v=torch.randn(_v.shape,generator=_g); _v=_v/_v.norm()*RA[0].norm()   # blockrand keeps the real +g and randomises only the REMOVED direction (D170B bug: it randomised both)
                _ps=[len(idl)-1] if _mode.startswith('q') else sorted({p for (s0,s1,_) in b.spans for p in range(label_lo(b,s0),s0)})
                if _mode.startswith('qplus'): _sgn=1.0
                H.DADD={RA[1]:(_v,torch.tensor(_ps),_sgn*RA[2])}
                if 'block' in _mode:   # remove the gate direction (or a random one of the same norm) from the named readers' query inputs: if +g rescues through the readers' queries, this must undo it
                    _gd=RA[0].float(); _gd=_gd/_gd.norm()
                    if _mode.endswith('rand'): _g=torch.Generator().manual_seed(5000+sid); _gd=torch.randn(_gd.shape,generator=_g); _gd=_gd/_gd.norm()
                    _kd={}; _bw=next((w_ for w_ in ('all','mlp','both') if _mode.startswith('qplusblock'+w_)),None); _pairs=hs; H.KDIRS_POS=None
                    if _bw is not None:
                        _a,_,_z=_mode[len('qplusblock'+_bw):].partition('-'); _Ls=list(range(int(_a),int(_z or _a)+1)); H.KDIRS_POS=[len(idl)-1]
                        _pairs=[(l_,h_) for l_ in _Ls for h_ in range(nh)] if _bw in ('all','both') else []
                        if _bw in ('mlp','both'):
                            for l_ in _Ls:
                                _n=getattr(H.layers[l_],'post_attention_layernorm',None); _w=_n.weight.detach().float().cpu() if _n is not None else torch.ones_like(_gd)
                                _kd.setdefault(l_,{})['mlp']=(F.normalize(_gd*_w,dim=0),None)
                    for l_,h_ in _pairs:
                        _n=getattr(H.layers[l_],'input_layernorm',None); _w=_n.weight.detach().float().cpu() if _n is not None else torch.ones_like(_gd)
                        _u=F.normalize(_gd*_w,dim=0); _kd.setdefault(l_,{}).setdefault('q',(_u,[]))[1].append(h_)
                    H.set_kdirs(_kd)
                if hl[0][0]=='rk': H.set_knock(conds[RKCOND] if RKCOND else hs)
            elif hl and hl[0][0]=='tp':
                assert hl[0][1] in SRCT, 'twin patch before its twin source'; H.clear()
                _tp,*_holds=hl[0][2].split('^'); _spec=twin_spec(build_twin_patch(_tp,SRCT[hl[0][1]],T),SRCT[hl[0][1]])
                for _hp in _holds:
                    assert SRCN is not None, 'hold part before the intact run'
                    for _l,_d in twin_spec(build_twin_patch(_hp,SRCN,T),SRCN).items():
                        if _l=='final': _spec['final']=_d
                        else: _spec.setdefault(_l,{}).update(_d)
                H.set_patch(_spec)
                _p=_tp.partition('@')[2] or None; H.PATCH_POS=None if len(hl[0])>3 else twin_positions(b,T,_p); H.PATCH_N=T+shared_prefix(b,b.twins[hl[0][1]])
            elif hl and hl[0][0]=='sv': H.clear(); H.SINKV=(set(hl[0][1]),hl[0][2],hl[0][3])
            elif hl and hl[0][0]=='cm': H.clear(); H.set_knock(hl[0][1]); H.MKNOCK=set(hl[0][2])   # D171 circuit-only: heads + whole MLP layers mean-ablated at KNOCK_POS
            if hl and hl[0][0]=='cm' and QOUT is not None:   # D173: resample ('rs') conditions take their reference from another battery prompt; others from the battery mean (or natural mean)
                if len(hl)>1 and hl[1][0]=='rs': _q=QOUT[RS_NEXT[sid]]; H.MEANO,H.MEANM=_q['o'],_q['m']
                elif 'circmean=kind' in EXTRAS: H.MEANO,H.MEANM=MEANO_KIND[b.kind],MEANM_KIND[b.kind]
                elif 'circmean=battery' in EXTRAS: H.MEANO,H.MEANM=MEANO_BAT,MEANM_BAT
                else: H.MEANO,H.MEANM=MEANO_NAT,MEANM_NAT
            elif hl and hl[0][0]=='hg':
                H.clear(); H.HGAIN={}
                for l_,h_ in hl[0][1]: H.HGAIN.setdefault(l_,{})[h_]=hl[0][2]
                H.set_knock(hl[0][3])
            elif not (hl and isinstance(hl[0][0],str) and hl[0][0]!='g'): H.set_knock(hl)   # plain head-knock conds only; typed conds ('tp','t','ra',...) were set above. The D173 'if cm and QOUT' line broke the elif chain so typed conds fell through here (crashed D188SG*/GD*)
            tgt=None; ids_in=ids
            if icl: tgt_,fb=target_ids(tok,idl,b.target); tgt=torch.tensor(tgt_); NMIS+=int(fb); ids_in=torch.cat([ids,tgt])
            if (RDUMP or HDUMP) and cname=='none' and b.spans: H.RESPOS=torch.tensor(sorted({p for (s0,s1,_) in b.spans for p in range(label_lo(b,s0),s1)})+[len(idl)-1]); H.RES={}   # last row = the query's final position (its arrow)
            if GP and H.RESPOS is None: H.RESPOS=torch.tensor([len(idl)-1]); H.RES={}
            o=run(ids_in); lp=F.log_softmax(o.logits[0].float(),-1); sp=surprisal(lp,ids)
            GPV={}
            if GP:
                for _l,_g in GP.items():
                    if _l in H.RES and H.RES[_l].shape[-1]==_g.shape[0]: _x=H.RES[_l][-1]; GPV[str(_l)]=round(float(_x@_g/_g.norm()),3)   # size guard: smoke models have a different hidden size than the 4B direction
                if not ((RDUMP or HDUMP) and cname=='none' and b.spans): H.RESPOS=None
            if (RDUMP or HDUMP) and cname=='none' and b.spans:
                _pos=H.RESPOS.tolist(); H.RESPOS=None
                np.savez(f'{OUT}/resid_{sid}.npz',pos=np.array(_pos),ids=np.array([idl[p] for p in _pos]),spans=np.array([[s0,s1] for (s0,s1,_) in b.spans]),kind=b.kind,target=str(b.target),text=tok.decode(idl),**{f'l{l}':H.RES[l].numpy() for l in RDUMP if l in H.RES},**{f'h{l}':H.oin[l][torch.tensor(_pos)].cpu().numpy() for l in HDUMP if l in H.oin})
            if hl and hl[0][0]=='t': SRCT[hl[0][1]]=H.snapshot()
            if TWINS and cname=='none': SRCN=H.snapshot()
            if GATEFN and cname=='none' and PROBE_HEADS: gate_functional_loupe(ids_in,T,b.spans,b.target,sid,b.kind)
            if TWINS and cname=='none' and PROBE_HEADS and any('houtj' in c_ for c_ in conds): JBASIS=grad_probe_basis(ids_in,T,b.spans,b.target)
            if TWINS and cname=='none' and PROBE_HEADS and any('houtr' in c_ for c_ in conds): _g=torch.Generator(device='cpu').manual_seed(1000+sid if isinstance(sid,int) else 1000); RBASIS=torch.linalg.qr(torch.randn(m.config.hidden_size,len(PROBE_HEADS),generator=_g))[0].to(dev)
            if hl and hl[0][0]=='tp': H.PATCH_N=None
            if PATCHH and cname=='none': CLEAN=H.snapshot()
            if PATCHH and cname==PSRC:
                SRC=H.snapshot()
                if PSRC_HEADS:   # removed write = clean attention-sublayer output minus knocked output, summed over the knocked layers
                    SRC['rem']=sum(CLEAN['aout'][l_]-SRC['aout'][l_] for l_ in sorted({l_ for l_,_ in PSRC_HEADS}))
            TG=flag_readout(H.kin,U,MU,SD,T); QA={}; QAH={}; AROW={}; SROW={}; TLP=TLP1=ALT=None; GEOM={}; NEURQ={}; PAIRW={}; DLA={}; DLAJ=0; MDLA={}; FRMS=None; LDIFF=None; DLAE=None; DLAREC=None; DLAFIN=None; hasalt_=False; NEUR={}; LSPLIT={}; VREAD={}
            if icl:
                AB=attn_by_layer(o,H.SOFT,L); AT={l:AB[l] for l in late}
                if TRACE_N and sid<TRACE_N and (cname=='none' or cname.startswith('circ_only')):   # D179 token observatory: every token through every layer and head, to a durable shard
                    import numpy as _np; os.makedirs(f'{OUT}/trace',exist_ok=True)
                    _hs=torch.stack([h[0].float() for h in o.hidden_states]).cpu()                      # [L+1,T,d] residual after each layer (0 = embeddings)
                    _att=_np.stack([AB[l].float().cpu().numpy() for l in sorted(AB)]).astype(_np.float16)   # [L,nh,T,T]
                    _wn=_np.stack([H.oin[l].float().view(H.oin[l].shape[0],nh,-1).norm(dim=-1).cpu().numpy() for l in sorted(H.oin)]).astype(_np.float16)   # [L,T,nh] per-head write norm (o_proj input) at every position
                    _pu=_np.stack([(_hs[l+1]@U[l].float().cpu()).numpy() for l in late])                 # [late,T] projection on the repeat-flag direction U_l
                    _pg=_np.stack([(_hs[l+1]@GV.float().cpu()).numpy() for l in range(L)]) if GV is not None else None   # [L,T] projection on the gate direction g
                    _np.savez_compressed(f'{OUT}/trace/{sid:03d}_{cname}.npz',hs=_hs.half().numpy(),att=_att,wnorm=_wn,proj_u=_pu,proj_g=_pg if _pg is not None else _np.zeros(0),late=_np.array(late),toks=_np.array(idl),tokstr=_np.array([tok.decode([t]) for t in idl]),spans=_np.array([(a,b_) for a,b_,*_ in b.spans]),kind=b.kind,target=str(b.target),cond=cname,T=T)
                if b.spans: QA=query_attention(AT,T,b.spans,tol=ATT_TOL); QAH=query_attention_heads(AT,T,b.spans,set(dup))
                if AROWH: AROW=query_row_top(AB,T,AROWH,idl,tok,k=AROWK)   # AB = every layer (AT only has the late ones; early heads' rows were empty before D178b)
                if SROWL and b.spans: SROW=slot_rows(AB,b.spans,idl,tok,SROWL)
                TLP,TLP1=seq_logprob(lp,T,tgt); TLP,TLP1=round(TLP,4),round(TLP1,4)
                alt_t=[]
                if b.alt:
                    alt_t,_=target_ids(tok,idl,b.alt)
                    if len(alt_t)==1: ALT=round(float(lp[T-1,alt_t[0]]),4)
                    else: H.PATCH_N=T; _st=H.stash(); o2=run(torch.cat([ids,torch.tensor(alt_t)])); H.unstash(_st); H.PATCH_N=None; lp2=F.log_softmax(o2.logits[0].float(),-1); ALT=round(seq_logprob(lp2,T,alt_t)[0],4)   # alt tail differs from the source run: patch the prompt prefix only
                if cname=='none' and b.spans:
                    # key/query geometry at each late layer: label keys' cos to U, to mean key, pairwise; q.u for the query per head; rank of label-key set
                    for l in late:
                        K=H.kin[l][:T].float(); lab=torch.stack([K[s0:s1].mean(0) for (s0,s1,_) in b.spans]); labn=F.normalize(lab,dim=-1); mk=F.normalize(K.mean(0),dim=0)
                        pw=labn@labn.T; iu=torch.triu_indices(len(lab),len(lab),1)
                        GEOM[str(l)]={'cos_u':[round(float(x),4) for x in (labn@U[l].to(labn.device)).tolist()],'cos_mean':[round(float(x),4) for x in (labn@mk).tolist()],
                                      'pair_cos_mean':round(float(pw[iu[0],iu[1]].mean()),4),'rank_90':int((torch.linalg.svdvals(lab-lab.mean(0)).cumsum(0)/torch.linalg.svdvals(lab-lab.mean(0)).sum()<0.9).sum()+1),
                                      'knorm':[round(float(x),3) for x in lab.norm(dim=-1).tolist()]}
                    # loupes: (i) MLP neurons writing onto the flag direction at each label position; (ii) pre-norm attention-logit split (along flag dir vs rest) from the query to each label, per head;
                    # (iii) value readout: what attending to each label would move along (target - alt) unembedding, per head
                    dvec=Wu[tgt[0]].detach().float()-(Wu[alt_t[0]].detach().float() if (b.alt and len(alt_t)) else 0); lab0=[s0 for (s0,s1,_) in b.spans]
                    torch.set_grad_enabled(False)   # weight tensors below carry requires_grad; no graph wanted
                    for (pl_,ph_) in PAIRW_HEADS:
                        if pl_ in H.oin:
                            Wo_=out_proj_matrix(H.layers[pl_]); hd__=Wo_.shape[0]//nh; o_=H.oin[pl_][T-1].to(Wo_.device)   # Wo_: [nh*hd, d]
                            PAIRW[f'{pl_}.{ph_}']=[round(float(x),4) for x in (o_[ph_*hd__:(ph_+1)*hd__]@Wo_[ph_*hd__:(ph_+1)*hd__]).tolist()]
                    for l in late:
                        if l in H.mlpin:
                            nw=mlp_neuron_writers(H.mlpin[l][:T],down_matrix(H.layers[l]),U[l],lab0,k=5); NEUR[str(l)]={str(p_):v for p_,v in nw.items()}
                            dq=dvec/dvec.norm() if torch.is_tensor(dvec) else None
                            if dq is not None:
                                wq=(dq.to(H.mlpin[l].device)@down_matrix(H.layers[l])); cq=H.mlpin[l][T-1].to(wq.device)*wq; tq=torch.topk(cq.abs(),8)
                                g_=H.gate.get(l); u_=H.up.get(l)
                                NEURQ[str(l)]={'total':round(float(cq.sum()),4),'top':[(int(i),round(float(cq[i]),4),(round(float(g_[T-1,i]),3) if g_ is not None else None),(round(float(u_[T-1,i]),3) if u_ is not None else None)) for i in tq.indices]}
                        q,k,v=qkv_pre(H,l); uk=key_dir(H,l,U[l]); grp=nh//H.nkv; Wo=out_proj_matrix(H.layers[l]); hd_=Wo.shape[0]//nh; sc=hd_**-0.5
                        al=[];rs=[];vr=[]
                        for h in range(nh):
                            kv=h//grp; tot,a_u,rest=attn_logit_split(q[T-1,h],k[:T,kv],uk[kv],sc); al.append([round(float(a_u[p_]),3) for p_ in lab0]); rs.append([round(float(rest[p_]),3) for p_ in lab0])
                            vr.append([round(float(x),3) for x in value_readout(v[:T,kv],Wo[h*hd_:(h+1)*hd_],dvec)[lab0].tolist()])
                        LSPLIT[str(l)]={'along_u':al,'rest':rs}; VREAD[str(l)]=vr
                # per-head direct logit attribution at the query to the target's first token (minus alt if present), via o_proj/c_proj weights and unembedding;
                # recorded intact AND under direction conditions so the heads whose direct write changes under flag removal can be named
                if (cname=='none' or cname.startswith('radd') or cname.startswith('dir_') or cname.startswith('kpath_') or cname.startswith('patch_') or cname.startswith('neur') or cname.startswith('tp_') or cname.startswith('twin_') or (PATCHH and cname==PSRC) or (NEURIN and cname=='lookback16')) and b.spans:
                    for l in H.SOFT:
                        oi=H.oin.get(l)
                        if oi is None: continue
                        W=out_proj_matrix(H.layers[l]); hd_=W.shape[0]//nh
                        # direction = target minus alternative at the FIRST token where they differ (digit labels ' 7' vs ' 3' share the leading space token, so index 0 would give d=0)
                        hasalt=bool(b.alt and len(alt_t)); hasalt_=hasalt; dj=next((i for i in range(min(len(tgt),len(alt_t))) if int(tgt[i])!=int(alt_t[i])),0) if hasalt else 0
                        if oi.shape[0]<=T-1+dj: continue
                        d=Wu[tgt[dj]].float()-(Wu[alt_t[dj]].float() if hasalt else 0); d=d*FNW.to(d.device); v=(W@d.to(W.device)).view(nh,hd_); DLAJ=dj   # D159: final-norm weight folded in, so sum(DLA)+sum(MDLA)+embed·d == ldiff·frms exactly (RMSNorm has no bias)
                        DLA[str(l)]=[round(float(x),3) for x in torch.einsum('hd,hd->h',oi[T-1+dj].view(nh,hd_).to(v.device),v).tolist()]
                        # same direction, per-layer MLP direct push (+ its top neurons) so the late-MLP route can be read next to the heads
                        hm=H.mlpin.get(l)
                        if hm is not None and hm.shape[0]>T-1+dj:
                            tot,top=mlp_direct(hm[T-1+dj],down_matrix(H.layers[l]).to(hm.device),d.to(hm.device)); MDLA[str(l)]={'tot':round(tot,3),'top':top}
                    # final-norm loupe: RMS of the final pre-norm residual at the direct-logit position and the true target-minus-alternative logit there,
                    # so a condition's change in residual scale (which rescales EVERY direct push) can be separated from changes in the pushes themselves
                    if H.fin is not None and H.fin.shape[0]>T-1+DLAJ and hasalt_:
                        f_=H.fin[T-1+DLAJ].float(); FRMS=round(float(f_.pow(2).mean().sqrt()),3); LDIFF=round(float(o.logits[0,T-1+DLAJ,int(tgt[DLAJ])]-o.logits[0,T-1+DLAJ,int(alt_t[DLAJ])]),3)
                        _d=(Wu[int(tgt[DLAJ])].float()-Wu[int(alt_t[DLAJ])].float())*FNW.to(Wu.device); _e=m.get_input_embeddings().weight[int(ids_in[T-1+DLAJ])].float().to(_d.device)
                        DLAE=round(float(_e@_d),3); _att=sum(sum(v) for v in DLA.values()); _mlp=sum(v['tot'] for v in MDLA.values()); DLAREC=round(float((_att+_mlp+DLAE)/max(FRMS,1e-6)),3); DLAFIN=round(float((f_.to(_d.device)@_d)/max(FRMS,1e-6)),3)   # reconstruction: heads+MLPs+embedding vs the exact final-residual projection vs the true logit diff
            TOP5=None
            try:
                _lp=torch.log_softmax(o.logits[0,T-1].float(),-1); _v,_i=_lp.topk(5); TOP5=[(tok.decode([int(i_)]),round(float(v_),3)) for v_,i_ in zip(_v,_i)]
            except Exception: pass
            fk.write(json.dumps({'seq':sid,'kind':b.kind,'cond':cname,'dup':dup,'alt_tlp':ALT,'gproj':GPV,'surp':[round(x,3) for x in sp.tolist()],'tag':TG,'qatt':QA,'qatth':QAH,'arow':AROW,'srow':SROW,'tlp':TLP,'tlp1':TLP1,
                                 'ntgt':(len(tgt) if tgt is not None else 0),'tgt_ids':(tgt.tolist() if tgt is not None else None),'tgt_dec':(tok.decode(tgt) if tgt is not None else None),
                                 'spans':[(s0,s1) for (s0,s1,_) in b.spans],'alt':b.alt,'target':b.target,'meta':b.meta,'geom':GEOM,'neurq':NEURQ,'pairw':PAIRW,'T':T,'dla':DLA,'dlaj':DLAJ,'mdla':MDLA,'frms':FRMS,'ldiff':LDIFF,'dlae':DLAE,'dlarec':DLAREC,'dlafin':DLAFIN,'neur':NEUR,'lsplit':LSPLIT,'vread':VREAD,'top5':TOP5})+'\n')
        b=b0; H.clear(); fk.flush(); print('knock seq',sid,b.kind,round(time.time()-t0),flush=True); status({'phase':'knock','seq':sid,'n':len(B),'elapsed':time.time()-t0})
status({'phase':'knock_done','elapsed':time.time()-t0,'target_prefix_mismatches':NMIS}); print('target prefix mismatches',NMIS,flush=True)
json.dump({'late':late,'nh':nh,'RO':RO,'MU':MU,'SD':SD,'split_cos':{str(l):float(UA[l]@UB[l]) for l in late}},open(f'{OUT}/run_summary.json','w'))
exp_kinds=','.join([k for f,k in [('std','icl_'),('redund','icl_redund'),('correct','icl_correct'),('bind','icl_bind'),('ctl','repeat_noun'),('labelswap','icl_labelswap'),('markcorrect','icl_markcorrect'),('concept','icl_concept'),('concept12','icl_concept'),
    ('slotgatex','icl_slotgate'),('functask','icl_functask'),('functask2','icl_functask2'),('functask3','icl_functask3'),('functask4','icl_functask4'),('functask5','icl_functask5')] if f in FAMS])
r=subprocess.run([sys.executable,f'{ROOT}/check.py',OUT,exp_kinds],capture_output=True,text=True); print(r.stdout[-3000:],r.stderr[-2000:],flush=True)
status({'phase':'done' if r.returncode==0 else 'CHECK_FAILED','elapsed':time.time()-t0,'check_rc':r.returncode}); print('run done',round(time.time()-t0),'check rc',r.returncode,flush=True)
