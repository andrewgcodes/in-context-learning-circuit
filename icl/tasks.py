import itertools
import os
"""Battery construction as data. One entry point: build_battery(tok, families, smoke) -> list of Item.
Families: 'std' (6-shot icl_<task> + 12-shot icl12_<task> over the learnable task list), 'redund', 'correct', 'bind', 'ctl'.
Families ADD rows; nothing replaces anything. Asserts every requested family produced rows and prints the kinds list.
Prompt format is byte-identical to the legacy grail_gpu.py tagscope builder (`x -> y\n` lines, query `x ->`)."""
import json,os,random,torch
from dataclasses import dataclass,field
from tokspans import label_spans,target_ids
ROOT=os.path.dirname(os.path.abspath(__file__))
def _battery(name):
    here=os.path.dirname(os.path.abspath(__file__))
    for d in (here, os.path.join(here,'..','batteries'), ROOT):
        p=os.path.join(d,name)
        if os.path.exists(p): return p
    raise FileNotFoundError(f'{name}: put it next to this file or in ../batteries')

LEARN_TASKS=lambda T:{k:v for k,v in T.items() if k.startswith(('cipher_k','mod_','novelword','sentiment')) or k in ('antonym','capital','past','upper','randmap_multi')}
SMOKE_TASKS=['antonym','mod_add','sentiment_num','novelword_dax_blick']

@dataclass
class Item:
    kind:str; ids:torch.Tensor; spans:list=field(default_factory=list); target:str=None; alt:str=None; meta:dict=field(default_factory=dict); twins:dict=field(default_factory=dict)   # name -> Item of identical length/label positions

def build(tok,pairs,query):
    text='';labpos=[]
    for x,y in pairs:
        pre=tok(text+x+' ->')['input_ids']; text=text+x+' -> '+y+'\n'; labpos.append(len(pre))
    q=tok(text+query+' ->')['input_ids']; return torch.tensor(q),labpos,len(q)-1

def mk(tok,kind,pairs,query,target,alt=None,meta=None):
    ids,labpos,_=build(tok,pairs,query); spans=label_spans(tok,ids.tolist(),pairs,labpos)
    target_ids(tok,ids.tolist(),target)   # assert the answer is scorable now, not at run time
    return Item(kind,ids,spans,target,alt,meta or {})
def other_label(pairs,target):
    """For 2-label tasks: the label that is not the target (scored as 'alt' so P(target) vs P(other) is read directly); None otherwise."""
    labs=sorted({y for _,y in pairs}); return labs[1-labs.index(target)] if len(labs)==2 and target in labs else None

def fam_std(tok,smoke):
    T=json.load(open(_battery('icl_tasks.json'))); T=LEARN_TASKS(T)
    if smoke: T={k:T[k] for k in SMOKE_TASKS}
    out=[]; random.seed(5)
    for task,pairs in T.items():
        for s_ in range(1 if smoke else 12):
            d=random.sample(pairs,7); out.append(mk(tok,f'icl_{task}',d[:6],d[6][0],d[6][1],alt=other_label(d[:6],d[6][1])))
        for s_ in range(1 if smoke else 4):
            k_=min(12,len(pairs)-1); d=random.sample(pairs,k_+1); out.append(mk(tok,f'icl12_{task}',d[:k_],d[k_][0],d[k_][1],alt=other_label(d[:k_],d[k_][1])))
    return out

SYN={'happy':'glad','large':'big','quick':'fast','cold':'chilly','angry':'mad','small':'little','smart':'clever','rich':'wealthy','tired':'sleepy','brave':'bold','quiet':'silent','loud':'noisy'}
def fam_redund(tok,smoke):
    out=[]; random.seed(9); base=[(k,'alpha') for k in list(SYN)[:6]]+[(k,'beta') for k in list(SYN)[6:]]
    for s_ in range(1 if smoke else 16):
        d=random.sample(base,7); xq,yq=d[6]
        for var in ('exact','para','ctrl'):
            dd=list(d[:6]); x1,y1=dd[0]
            if var=='exact': dd[2]=(x1,y1)
            elif var=='para': dd[2]=(SYN[x1],y1)
            out.append(mk(tok,f'icl_redund_{var}',dd,xq,yq,alt=other_label(dd,yq),meta={'labels':[y for _,y in dd]}))
    return out

NOUNS2=['cactus','button','helmet','tunnel','river','pillow','desert','ladder','candle','anchor','marble','violin']
COLS2=['crimson','olive','navy','amber','teal','coral','ivory','plum']
def fam_correct(tok,smoke):
    out=[]; random.seed(10)
    for s_ in range(1 if smoke else 16):
        ns=random.sample(NOUNS2,7); cs=random.sample(COLS2,7); d=list(zip(ns,cs)); x1,B_=d[0]; A_=random.choice([c for c in COLS2 if c!=B_])
        for var in ('correct','confirm','ctrl'):
            dd=list(d[:6])
            if var=='correct': dd[0]=(x1,A_); dd[3]=(x1,B_)
            elif var=='confirm': dd[0]=(x1,B_); dd[3]=(x1,B_)
            else: dd[0]=d[6]; dd[3]=(x1,B_)
            out.append(mk(tok,f'icl_correct_{var}',dd,x1,B_,alt=A_))
    return out

CITIES=['Paris','Rome','Tokyo','Madrid','Berlin','Lisbon','Oslo','Cairo']; NAMES=['Alice','Bob','Carol','Dave','Erin','Frank','Grace','Heidi']
def fam_bind(tok,smoke):
    out=[]; random.seed(11)
    for s_ in range(1 if smoke else 16):
        ns=random.sample(NAMES,4); cs=random.sample(CITIES,4); q=random.randrange(4)
        for var in ('single','double'):
            sents=[]
            for i,(n,c) in enumerate(zip(ns,cs)):
                sents.append(f'{n} lives in {c}.')
                if var=='double' and i==q: sents.append(f'{n} works as a teacher.')
            txt=' '.join(sents)+f' Question: where does {ns[q]} live? Answer:'
            ids=torch.tensor(tok(txt)['input_ids']); target_ids(tok,ids.tolist(),cs[q])
            out.append(Item(f'icl_bind_{var}',ids,[],cs[q]))
    return out

NOUNS=['apple','river','stone','cloud','piano','tiger','candle','window','garden','mirror','pencil','rocket','castle','violin','basket','meadow']
def fam_ctl(tok,smoke):
    out=[]
    for n_ in NOUNS[:(2 if smoke else 16)]: out.append(Item('repeat_noun',torch.tensor(tok(' '.join([n_]*12))['input_ids'])))
    V=min(tok.vocab_size,50000); rng=random.Random(21)
    for sd_ in range(1 if smoke else 20):
        rng.seed(100+sd_); base=torch.tensor(rng.sample(range(1000,V-1000),40)); g=torch.Generator().manual_seed(200+sd_)
        out.append(Item('random_tokens_repeated',torch.cat([base,base]))); out.append(Item('random_tokens_shuffled',torch.cat([base,base[torch.randperm(40,generator=g)]])))
    return out

LABELSETS={'sym':('foo','bar'),'num':('7','3'),'word':('good','bad'),'rare':('zorp','quil'),'letter':('A','B'),'symR':('bar','foo'),'wordR':('bad','good'),'rareR':('quil','zorp')}   # *R = same labels with the class roles reversed (breaks the class/label-identity tie across families)
BLANK={'sym':['xyz','qqq','zzz'],'num':['0','5'],'word':['thing','item'],'rare':['zzz'],'letter':['X','Z']}   # neutral same-length label candidates
THIRD={'sym':[('baz','fizz'),('baz','zap'),('qux','baz')],'num':[('8','2')],'word':[('nice','poor')],'rare':[('blik','dax'),('vorp','glim')],'letter':[('C','D')]}   # candidate relabel pairs; first whose token lengths match the original pair is used
def add_twins(tok,item,demos,q,qpos,lp,ln,name):
    """Attach counterfactual twins to a labelswap item: 'swap' = same prompt with the two labels' roles exchanged (target becomes the alt),
    'relabel' = same mapping with a third label pair. Kept only when the twin tokenizes to the same length with label spans at the same positions."""
    def mk_twin(a,b):
        dd=[(x,a if y=='P' else b) for x,y in demos]; t=mk(tok,item.kind,dd,q,a if qpos else b,alt=(b if qpos else a),meta=dict(item.meta))
        ok=len(t.ids)==len(item.ids) and [(s0,s1) for s0,s1,_ in t.spans]==[(s0,s1) for s0,s1,_ in item.spans] and len(target_ids(tok,t.ids.tolist(),t.target)[0])==len(target_ids(tok,item.ids.tolist(),item.target)[0])
        return t if ok else None
    t=mk_twin(ln,lp)
    if t is not None: item.twins['swap']=t
    # 'shuffle': same label tokens at the same positions but the mapping is inconsistent (first half of the demos, in prompt order, use the
    # swapped assignment) -> no learnable rule; 'blank': every label replaced by one neutral token of the same length -> nothing to retrieve.
    # Both score the item's OWN target, so they are sources only (their rows show how the model does without a usable context).
    def mk_var(labels,tgt,alt):
        pairs=[(x,y) for (x,_),y in zip(demos,labels)]; t=mk(tok,item.kind,pairs,q,tgt,alt=alt,meta=dict(item.meta))
        ok=len(t.ids)==len(item.ids) and [(s0,s1) for s0,s1,_ in t.spans]==[(s0,s1) for s0,s1,_ in item.spans]
        return t if ok else None
    half=len(demos)//2; sh=[(ln if y=='P' else lp) if i<half else (lp if y=='P' else ln) for i,(_,y) in enumerate(demos)]
    t=mk_var(sh,item.target,item.alt)
    if t is not None: item.twins['shuffle']=t
    # 'task': same tokens and label multiset, but a different CONSISTENT rule — label by demo slot parity (even slots lp, odd slots ln);
    # the query is slot 6 (even) so its target under the new rule is lp. Tests whether a reader carries the TASK rather than the labels.
    t=mk_var([lp if i%2==0 else ln for i in range(len(demos))],lp,ln)
    if t is not None: item.twins['task']=t
    # 'sent': the FIRST demo whose class matches the query keeps its label but gets an unused opposite-class sentence of the same token length
    # -> that slot's VALUE (label) is unchanged, only the class of its sentence (the hypothesised KEY) flips. Scores the item's own target (source twin).
    pool=SENTPOOL.get('pool'); used={x for x,_ in demos}|{q}
    if pool is not None:
        for k_,(x,y) in enumerate(demos):
            if (y=='P')!=qpos: continue
            n_=len(tok(x)['input_ids']); cands=[z for z,c in pool if c!=y and z not in used and len(tok(z)['input_ids'])==n_]
            if cands:
                t=mk(tok,item.kind,[(cands[0] if j==k_ else dx, dl) for j,(dx,dl) in enumerate([(dx,lp if dy=='P' else ln) for dx,dy in demos])],q,item.target,alt=item.alt,meta={**item.meta,'sent_slot':k_})
                if len(t.ids)==len(item.ids) and [(s0,s1) for s0,s1,_ in t.spans]==[(s0,s1) for s0,s1,_ in item.spans]: item.twins['sent']=t
            break
    # 'para': SAME-ANSWER control (Quirke 2606.08292) — first query-class demo gets an unused SAME-class sentence of equal token length;
    # key, value and answer unchanged. Head/slot handoffs from this twin should move ~0 if they carry the computation, not generic prompt state.
    if pool is not None:
        for k_,(x,y) in enumerate(demos):
            if (y=='P')!=qpos: continue
            n_=len(tok(x)['input_ids']); cands=[z for z,c in pool if c==y and z not in used and len(tok(z)['input_ids'])==n_]
            if cands:
                t=mk(tok,item.kind,[(cands[0] if j==k_ else dx, dl) for j,(dx,dl) in enumerate([(dx,lp if dy=='P' else ln) for dx,dy in demos])],q,item.target,alt=item.alt,meta={**item.meta,'para_slot':k_})
                if len(t.ids)==len(item.ids) and [(s0,s1) for s0,s1,_ in t.spans]==[(s0,s1) for s0,s1,_ in item.spans]: item.twins['para']=t
            break
    # 'sentx': EXCHANGE the sentences of one same-class and one opposite-class demo (equal token length), labels stay in place
    # -> two slots now have (key=class of new sentence, value=old label). Key hypothesis + Bayes: query attends to the 3 query-class
    # sentences, 2 right / 1 wrong label -> about log(2/3) = -0.41 nats; if the fetch ignores sentences: ~0.
    dl_=[(dx,lp if dy=='P' else ln) for dx,dy in demos]; done=False
    for i_,(xi,yi) in enumerate(demos):
        if done or (yi=='P')!=qpos: continue
        for j_,(xj,yj) in enumerate(demos):
            if yj==yi or len(tok(xi)['input_ids'])!=len(tok(xj)['input_ids']): continue
            d2=list(dl_); d2[i_]=(xj,dl_[i_][1]); d2[j_]=(xi,dl_[j_][1])
            t=mk(tok,item.kind,d2,q,item.target,alt=item.alt,meta={**item.meta,'sentx_slots':[i_,j_]})
            if len(t.ids)==len(item.ids) and [(s0,s1) for s0,s1,_ in t.spans]==[(s0,s1) for s0,s1,_ in item.spans]: item.twins['sentx']=t; done=True; break
    # 'qx': QUERY-class flip — demos identical, the query sentence replaced by an unused OPPOSITE-class sentence of equal token length;
    # the twin's own target/alt are exchanged. Patching the twin's query-position activations into the item asks which components carry the query's class.
    if pool is not None:
        nq=len(tok(q)['input_ids']); qc='P' if qpos else 'N'
        for z,c in pool:
            if c==qc or z in used or len(tok(z)['input_ids'])!=nq: continue
            t=mk(tok,item.kind,dl_,z,item.alt,alt=item.target,meta={**item.meta,'qx':True})
            if len(t.ids)==len(item.ids) and [(s0,s1) for s0,s1,_ in t.spans]==[(s0,s1) for s0,s1,_ in item.spans]: item.twins['qx']=t; break
    for bl in BLANK.get(name,[]):
        t=mk_var([bl]*len(demos),item.target,item.alt)
        if t is not None: item.twins['blank']=t; break
    for a,b in THIRD.get(name,[]):
        t=mk_twin(a,b)
        if t is not None: item.twins['relabel']=t; break
SENTPOOL={}
def fam_labelswap(tok,smoke):
    """Same sentiment sentences, five label vocabularies; 6-shot, 3 pos + 3 neg demos in random order; query = held-out sentence. Reads whether the look-back dependence is a property of the label tokens."""
    T=json.load(open(_battery('icl_tasks.json')))['sentiment_sym']; pos=[x for x,y in T if y=='foo']; neg=[x for x,y in T if y=='bar']
    SENTPOOL['pool']=[(x,'P') for x in pos]+[(x,'N') for x in neg]
    out=[]; random.seed(12)
    for s_ in range(1 if smoke else 24):
        ps=random.sample(pos,4); ns=random.sample(neg,4); qpos=random.random()<0.5
        demos=[(x,'P') for x in ps[:3]]+[(x,'N') for x in ns[:3]]; random.shuffle(demos); q=(ps[3] if qpos else ns[3])
        for name,(lp,ln) in LABELSETS.items():
            dd=[(x,lp if y=='P' else ln) for x,y in demos]; tgt=lp if qpos else ln
            it=mk(tok,f'icl_labelswap_{name}',dd,q,tgt,alt=(ln if qpos else lp),meta={'labelset':name}); add_twins(tok,it,demos,q,qpos,lp,ln,name); out.append(it)
    return out
ANTONYMS=[('hot','cold'),('big','small'),('up','down'),('fast','slow'),('happy','sad'),('old','new'),('light','dark'),('open','closed'),('rich','poor'),('wet','dry'),
 ('early','late'),('hard','soft'),('high','low'),('long','short'),('loud','quiet'),('strong','weak'),('thick','thin'),('clean','dirty'),('full','empty'),('near','far'),
 ('sweet','sour'),('tall','short'),('wide','narrow'),('young','old'),('push','pull'),('buy','sell'),('win','lose'),('love','hate'),('start','stop'),('day','night')]
def fam_func(tok,smoke):
    """FUNCTION in-context learning: 6-shot antonym demos 'x -> y', query word whose antonym is NOT in the context. The answer must be computed, not
    retrieved. Twins: 'blank' (every demo output replaced by the same neutral token: no examples of the function) and 'task' (same input words, outputs
    follow a different rule: the input word itself = identity/copy task; target then = the query word). Both keep token lengths/spans when possible."""
    out=[]; random.seed(21)
    for s_ in range(1 if smoke else 24):
        pairs=random.sample(ANTONYMS,7); demos=pairs[:6]; qx,qy=pairs[6]
        it=mk(tok,'icl_func_antonym',demos,qx,qy,alt=qx,meta={'q':qx})
        dd_blank=[(x,'thing') for x,_ in demos]; t=mk(tok,'icl_func_antonym',dd_blank,qx,qy,alt=qx,meta={'q':qx})
        if len(t.ids)==len(it.ids) and [(a,b) for a,b,_ in t.spans]==[(a,b) for a,b,_ in it.spans]: it.twins['blank']=t
        dd_copy=[(x,x) for x,_ in demos]; t=mk(tok,'icl_func_antonym',dd_copy,qx,qx,alt=qy,meta={'q':qx})
        if len(t.ids)==len(it.ids) and [(a,b) for a,b,_ in t.spans]==[(a,b) for a,b,_ in it.spans]: it.twins['task']=t
        out.append(it)
    return out

def fam_functask(tok,smoke):
    """Rule-direction battery (D41): the same 6 demo inputs under two consistent rules, each as its own item so its slots get dumped:
    'anto' (x -> antonym, query answered by the antonym) and 'copy' (x -> x, query answered by the word itself). Each item's 'task' twin is the
    other rule's prompt (same tokens/spans when possible), so the rule direction = anto-slot minus copy-slot residual at the arrows."""
    out=[]; random.seed(21)
    for s_ in range(1 if smoke else 24):
        pairs=random.sample(ANTONYMS,7); demos=pairs[:6]; qx,qy=pairs[6]
        dd_copy=[(x,x) for x,_ in demos]
        a=mk(tok,'icl_functask_anto',demos,qx,qy,alt=qx,meta={'q':qx,'rule':'anto'})
        c=mk(tok,'icl_functask_copy',dd_copy,qx,qx,alt=qy,meta={'q':qx,'rule':'copy'})
        ta=mk(tok,'icl_functask_anto',dd_copy,qx,qy,alt=qx,meta={'q':qx,'rule':'anto'})
        tc=mk(tok,'icl_functask_copy',demos,qx,qx,alt=qy,meta={'q':qx,'rule':'copy'})
        same=lambda t,it: len(t.ids)==len(it.ids) and [(p,q) for p,q,_ in t.spans]==[(p,q) for p,q,_ in it.spans]
        if same(ta,a): a.twins['task']=ta
        if same(tc,c): c.twins['task']=tc
        out+=[a,c]
    return out

def fam_functask2(tok,smoke):
    """D45: (i) a SECOND function pair with the same construction as fam_functask: 'cap' (country -> capital) vs 'copy' (country -> country), same 6 demo
    countries, 'task' twin = the other rule; (ii) 1-shot and 2-shot ANTONYM items (no twin) to ask whether the antonym heads fire and attend to the query word
    with almost no demonstrations (pretrained function heads) — kinds icl_functask2_anto1 / _anto2 with alt = the query word (copy answer)."""
    out=[]; random.seed(23)
    for s_ in range(1 if smoke else 24):
        pairs=random.sample(CAPITALS,7); demos=[(c,k) for c,k,_ in pairs[:6]]; qx,qy,_=pairs[6]
        dd_copy=[(x,x) for x,_ in demos]
        a=mk(tok,'icl_functask2_cap',demos,qx,qy,alt=qx,meta={'q':qx,'rule':'cap'})
        c=mk(tok,'icl_functask2_copy',dd_copy,qx,qx,alt=qy,meta={'q':qx,'rule':'copy'})
        ta=mk(tok,'icl_functask2_cap',dd_copy,qx,qy,alt=qx,meta={'q':qx,'rule':'cap'})
        tc=mk(tok,'icl_functask2_copy',demos,qx,qx,alt=qy,meta={'q':qx,'rule':'copy'})
        same=lambda t,it: len(t.ids)==len(it.ids) and [(p,q) for p,q,_ in t.spans]==[(p,q) for p,q,_ in it.spans]
        if same(ta,a): a.twins['task']=ta
        if same(tc,c): c.twins['task']=tc
        out+=[a,c]
        ap=random.sample(ANTONYMS,3); ax,ay=ap[2]
        for n_ in (1,2):   # self-twin: run.py keeps only items that have the requested twin; the twin conds are then no-ops for these read-only items
            it=mk(tok,f'icl_functask2_anto{n_}',ap[:n_],ax,ay,alt=ax,meta={'q':ax,'rule':f'anto{n_}'}); it.twins['task']=it; out.append(it)
        cp=random.sample(CAPITALS,3); cx,cy,_=cp[2]
        for n_ in (1,2):
            it=mk(tok,f'icl_functask2_cap{n_}',[(c,k) for c,k,_ in cp[:n_]],cx,cy,alt=cx,meta={'q':cx,'rule':f'cap{n_}'}); it.twins['task']=it; out.append(it)
    return out

# second FUNCTION task: country -> capital (answer absent from the context, not a copy of anything in it). Twins: 'blank' (outputs = 'thing'),
# 'task' (same countries, outputs follow a different rule: the country's language; target then = the query country's language, alt = its capital).
CAPITALS=[('France','Paris','French'),('Spain','Madrid','Spanish'),('Italy','Rome','Italian'),('Germany','Berlin','German'),('Japan','Tokyo','Japanese'),
 ('China','Beijing','Chinese'),('Russia','Moscow','Russian'),('Egypt','Cairo','Arabic'),('Greece','Athens','Greek'),('Portugal','Lisbon','Portuguese'),
 ('Turkey','Ankara','Turkish'),('Poland','Warsaw','Polish'),('Sweden','Stockholm','Swedish'),('Norway','Oslo','Norwegian'),('Finland','Helsinki','Finnish'),
 ('Hungary','Budapest','Hungarian'),('Austria','Vienna','German'),('Ireland','Dublin','English'),('Thailand','Bangkok','Thai'),('Vietnam','Hanoi','Vietnamese'),
 ('Iran','Tehran','Persian'),('Kenya','Nairobi','Swahili'),('Peru','Lima','Spanish'),('Cuba','Havana','Spanish'),('Denmark','Copenhagen','Danish'),
 ('Netherlands','Amsterdam','Dutch'),('Korea','Seoul','Korean'),('Mexico','Mexico City','Spanish'),('Canada','Ottawa','English'),('Chile','Santiago','Spanish')]
def fam_func2(tok,smoke):
    """FUNCTION in-context learning #2: 6-shot 'country -> capital', query country whose capital is NOT in the context."""
    out=[]; random.seed(23)
    for s_ in range(1 if smoke else 24):
        tr=random.sample(CAPITALS,7); demos=[(c,k) for c,k,_ in tr[:6]]; qc,qk,ql=tr[6]
        it=mk(tok,'icl_func_capital',demos,qc,qk,alt=ql,meta={'q':qc})
        dd_blank=[(c,'thing') for c,_ in demos]; t=mk(tok,'icl_func_capital',dd_blank,qc,qk,alt=ql,meta={'q':qc})
        if len(t.ids)==len(it.ids) and [(a,b) for a,b,_ in t.spans]==[(a,b) for a,b,_ in it.spans]: it.twins['blank']=t
        dd_lang=[(c,l) for c,_,l in tr[:6]]; t=mk(tok,'icl_func_capital',dd_lang,qc,ql,alt=qk,meta={'q':qc})
        if len(t.ids)==len(it.ids) and [(a,b) for a,b,_ in t.spans]==[(a,b) for a,b,_ in it.spans]: it.twins['task']=t
        out.append(it)
    return out
MARKERS={'bare':'{x} -> {y}','correction':'Correction: {x} -> {y}','no':'No, {x} -> {y}','actually':'Actually {x} -> {y}','update':'Update: {x} -> {y}'}
def fam_markcorrect(tok,smoke):
    """Correction with an explicit marker on the correcting line (slot 3): x1 -> A at slot 0, '<marker> x1 -> B' at slot 3, query x1. Target B, alt A. 'bare' reproduces fam_correct's 'correct' variant."""
    out=[]; random.seed(13)
    for s_ in range(1 if smoke else 16):
        ns=random.sample(NOUNS2,7); cs=random.sample(COLS2,7); d=list(zip(ns,cs)); x1,B_=d[0]; A_=random.choice([c for c in COLS2 if c!=B_])
        for mname,fmt in MARKERS.items():
            dd=list(d[:6]); dd[0]=(x1,A_); dd[3]=(x1,B_)
            # build text manually so the marker is part of line 3's input side; label spans come from the same builder as everything else
            pairs=[(fmt.format(x=x,y=y).rsplit(' -> ',1)[0],y) if i==3 else (x,y) for i,(x,y) in enumerate(dd)]
            out.append(mk(tok,f'icl_markcorrect_{mname}',pairs,x1,B_,alt=A_,meta={'marker':mname}))
    return out
CONCEPTS=json.load(open(_battery('concept_tasks.json')))   # concept -> (class-A sentences, class-B sentences)
CONCEPT_PER=3; CONCEPT_SUF=''; CONCEPT_ONLY=None
def fam_concept12(tok,smoke):
    """D79b: the no-world-knowledge axes (symset, cvc, digrule) at 12 shots (6 per class), same construction/twins as fam_concept."""
    global CONCEPT_PER,CONCEPT_SUF,CONCEPT_ONLY
    CONCEPT_PER,CONCEPT_SUF,CONCEPT_ONLY=6,'_12s',{'symset','cvc','digrule'}
    try: return fam_concept(tok,smoke)
    finally: CONCEPT_PER,CONCEPT_SUF,CONCEPT_ONLY=3,'',None
def fam_concept(tok,smoke):
    """New-concept battery: the same 6-shot 'sentence -> label' format, but the two classes are defined by a feature other than sentiment.
    topic (animals/food), lang (English/Spanish), question (question/statement) = features the encoder plausibly computes;
    parity (even/odd word count), vowel3 (third letter of the first word is a vowel) = features it plausibly does not.
    Demos are drawn so that at least one A and one B sentence share a token length (so the sentx twin exists)."""
    out=[]; random.seed(13+int(os.environ.get('BATTERY_SEED','0'))); LS={'sym':('foo','bar'),'rare':('zorp','quil')}
    for cname,(A,Bs) in CONCEPTS.items():
        if CONCEPT_ONLY and cname not in CONCEPT_ONLY: continue
        SENTPOOL['pool']=[(x,'P') for x in A]+[(x,'N') for x in Bs]
        for s_ in range(1 if smoke else 24):
            for _try in range(50):
                ps=random.sample(A,CONCEPT_PER+1); ns=random.sample(Bs,CONCEPT_PER+1)
                la={len(tok(x)['input_ids']) for x in ps[:3]}; lb={len(tok(x)['input_ids']) for x in ns[:3]}
                if la&lb: break
            qpos=random.random()<0.5; demos=[(x,'P') for x in ps[:CONCEPT_PER]]+[(x,'N') for x in ns[:CONCEPT_PER]]; random.shuffle(demos); q=(ps[CONCEPT_PER] if qpos else ns[CONCEPT_PER])
            if os.environ.get('CONCEPT_ORDER') in ('lastsame','lastother'):   # D128 recency test: move one demo of the query's class (lastsame) or the other class (lastother) to the END; everything else keeps its relative order
                _want=('P' if qpos else 'N') if os.environ['CONCEPT_ORDER']=='lastsame' else ('N' if qpos else 'P'); _i=max(i for i,d in enumerate(demos) if d[1]==_want); demos=demos[:_i]+demos[_i+1:]+[demos[_i]]
            if (os.environ.get('CONCEPT_ORDER') or '').startswith('sameat'):   # distance test: the query-class demos occupy positions (k-1,k) of the demo list (k=1..len-1), others keep their relative order -> the LAST same-class demo sits k positions from the start
                _k=int(os.environ['CONCEPT_ORDER'][6:]); _want='P' if qpos else 'N'; _same=[d for d in demos if d[1]==_want]; _oth=[d for d in demos if d[1]!=_want]
                _new=[None]*len(demos); _new[_k-1]=_same[0]; _new[_k]=_same[1]; _it=iter(_oth+_same[2:]); demos=[d if d is not None else next(_it) for d in _new]
            if (os.environ.get('CONCEPT_ORDER') or '').startswith('pattern'):   # label-sequence test: arrange demos by a pattern over S(ame class as query)/O(ther), e.g. pattern:SOSO; demos keep their relative order within each class
                _pat=os.environ['CONCEPT_ORDER'].split(':',1)[1]; _want='P' if qpos else 'N'; _S=iter([d for d in demos if d[1]==_want]); _O=iter([d for d in demos if d[1]!=_want]); assert len(_pat)==len(demos),(_pat,len(demos))
                demos=[next(_S) if ch=='S' else next(_O) for ch in _pat]
            if os.environ.get('CONCEPT_PERM'): _pr=random.Random(7919+s_); demos=sorted(demos,key=lambda _d:_pr.random())   # D127 order control: same sentences/labels/query, demonstrations re-ordered (seeded per seq, independent of the main RNG stream)
            for name,(lp,ln) in LS.items():
                dd=[(x,lp if y=='P' else ln) for x,y in demos]; tgt=lp if qpos else ln
                it=mk(tok,f'icl_concept_{cname}_{name}'+CONCEPT_SUF,dd,q,tgt,alt=(ln if qpos else lp),meta={'labelset':name,'concept':cname}); add_twins(tok,it,demos,q,qpos,lp,ln,name); out.append(it)
    return out


# third FUNCTION pair (D47): verb -> past tense vs verb -> verb (copy). Same construction as functask2; 'task' twin = the other rule.
PASTS=[('walk','walked'),('jump','jumped'),('play','played'),('open','opened'),('cook','cooked'),('wash','washed'),('push','pushed'),('pull','pulled'),
 ('start','started'),('finish','finished'),('watch','watched'),('climb','climbed'),('paint','painted'),('clean','cleaned'),('kick','kicked'),('laugh','laughed'),
 ('listen','listened'),('visit','visited'),('ask','asked'),('help','helped'),('work','worked'),('call','called'),('look','looked'),('want','wanted'),
 ('need','needed'),('turn','turned'),('learn','learned'),('fill','filled'),('pick','picked'),('talk','talked'),('sing','sang'),('run','ran'),('eat','ate'),
 ('go','went'),('see','saw'),('take','took'),('give','gave'),('write','wrote'),('drink','drank'),('swim','swam'),('fly','flew'),('break','broke')]
def fam_functask3(tok,smoke):
    """D47: third function pair 'past' (verb -> past tense) vs 'copy' (verb -> verb), same 6 demo verbs, 'task' twin = the other rule (exact tokens/spans required)."""
    out=[]; random.seed(29)
    for s_ in range(1 if smoke else 30):
        pairs=random.sample(PASTS,7); demos=pairs[:6]; qx,qy=pairs[6]
        dd_copy=[(x,x) for x,_ in demos]
        a=mk(tok,'icl_functask3_past',demos,qx,qy,alt=qx,meta={'q':qx,'rule':'past'})
        c=mk(tok,'icl_functask3_copy',dd_copy,qx,qx,alt=qy,meta={'q':qx,'rule':'copy'})
        ta=mk(tok,'icl_functask3_past',dd_copy,qx,qy,alt=qx,meta={'q':qx,'rule':'past'})
        tc=mk(tok,'icl_functask3_copy',demos,qx,qx,alt=qy,meta={'q':qx,'rule':'copy'})
        same=lambda t,it: len(t.ids)==len(it.ids) and [(p,q) for p,q,_ in t.spans]==[(p,q) for p,q,_ in it.spans]
        if same(ta,a): a.twins['task']=ta
        if same(tc,c): c.twins['task']=tc
        out+=[a,c]
        pp=random.sample(PASTS,3); px,py=pp[2]
        for n_ in (1,2):
            it=mk(tok,f'icl_functask3_past{n_}',pp[:n_],px,py,alt=px,meta={'q':px,'rule':f'past{n_}'}); it.twins['task']=it; out.append(it)
    return out


# D49: an UNOWNED function vs owned ones on the same inputs. 'rev' (word -> letters reversed; a rule the model cannot execute zero-shot at this size),
# 'upper' (word -> UPPERCASE; owned, Merullo 2023), 'copy'. Twins: each rule's 'task' twin is the copy prompt with the same inputs (exact tokens/spans required).
REVWORDS=['cat','dog','sun','map','pen','cup','hat','bed','car','bus','fan','jar','key','log','net','owl','pig','rat','tap','van','web','yam','zip','box','fig','gem','hen','ink','jam','kit','lip','mud','nut','oak','pot','rug','saw','tub','urn','vet','wax','yak','arm','bag','cow','den','egg','fox']
def fam_functask4(tok,smoke):
    """D49: 'rev' (x -> reversed letters), 'upper' (x -> X uppercased), 'copy' (x -> x) on the same 6 three-letter words; alt for rev/upper = the copy answer, alt for copy = the reversed word."""
    out=[]; random.seed(31)
    for s_ in range(1 if smoke else 30):
        ws=random.sample(REVWORDS,7); demos=ws[:6]; qx=ws[6]
        rv=lambda w: w[::-1]
        dd={'rev':[(w,rv(w)) for w in demos],'upper':[(w,w.upper()) for w in demos],'copy':[(w,w) for w in demos]}
        ans={'rev':rv(qx),'upper':qx.upper(),'copy':qx}
        c=mk(tok,'icl_functask4_copy',dd['copy'],qx,qx,alt=rv(qx),meta={'q':qx,'rule':'copy'})
        tc=mk(tok,'icl_functask4_copy',dd['rev'],qx,qx,alt=rv(qx),meta={'q':qx,'rule':'copy'})
        same=lambda t,it: len(t.ids)==len(it.ids) and [(p,q) for p,q,_ in t.spans]==[(p,q) for p,q,_ in it.spans]
        c.twins['task']=tc if same(tc,c) else c   # self-twin keeps read-only items (run.py drops items without the requested twin)
        out.append(c)
        for rule in ('rev','upper'):
            a=mk(tok,f'icl_functask4_{rule}',dd[rule],qx,ans[rule],alt=qx,meta={'q':qx,'rule':rule})
            ta=mk(tok,f'icl_functask4_{rule}',dd['copy'],qx,ans[rule],alt=qx,meta={'q':qx,'rule':rule})
            a.twins['task']=ta if same(ta,a) else a
            out.append(a)
    return out


# D54: a rule the model very likely has NO pretrained route for: 'sort' (word -> its letters in alphabetical order) on 4-letter words, with 'upper' as the owned control and 'copy'. Self-twins (read-only); directions fitted on CPU from rdump by query word.
SORTWORDS=['bark','cold','damp','fish','gold','hump','jump','kind','lamp','mind','nose','palm','rock','sand','tusk','vast','wolf','yarn','bolt','crab','dusk','fern','grim','husk','iron','junk','knob','loft','mask','nest','oval','pump','quiz','raft','silk','tomb','undo','vine','wisp','zinc','acid','bush','clip','drum','edge','frog','glum','hint']
def fam_functask5(tok,smoke):
    out=[]; random.seed(37)
    for s_ in range(1 if smoke else 30):
        ws=random.sample(SORTWORDS,7); demos=ws[:6]; qx=ws[6]; srt=lambda w: ''.join(sorted(w))
        dd={'sort':[(w,srt(w)) for w in demos],'upper':[(w,w.upper()) for w in demos],'copy':[(w,w) for w in demos]}
        ans={'sort':srt(qx),'upper':qx.upper(),'copy':qx}
        for rule in ('copy','sort','upper'):
            alt=srt(qx) if rule!='sort' else qx
            it=mk(tok,f'icl_functask5_{rule}',dd[rule],qx,ans[rule],alt=alt,meta={'q':qx,'rule':rule}); it.twins['task']=it; out.append(it)
    return out

def fam_functask5p(tok,smoke):
    """D54d: tokenization-matched unowned-function test. Same words/demos as functask5, but the SORT items' alternative is a random permutation of the query's letters
    (not sorted, not the word) with the SAME token count as the sorted string; copy items keep alt=sorted. If the model has no sorting ability, lp(sorted) ~ lp(perm)."""
    out=[]; random.seed(37); srt=lambda w: ''.join(sorted(w)); ntok=lambda s: len(tok(' '+s)['input_ids'])
    for s_ in range(1 if smoke else 30):
        ws=random.sample(SORTWORDS,7); demos=ws[:6]; qx=ws[6]
        perms=[''.join(p) for p in itertools.permutations(qx)]; perms=[p for p in perms if p!=srt(qx) and p!=qx and ntok(p)==ntok(srt(qx))]
        rs=random.Random(100+s_); rs.shuffle(perms)
        if not perms: continue
        dd={'sort':[(w,srt(w)) for w in demos],'copy':[(w,w) for w in demos]}
        it=mk(tok,'icl_functask5p_sort',dd['sort'],qx,srt(qx),alt=perms[0],meta={'q':qx,'rule':'sort','perm':perms[0]}); it.twins['task']=it; out.append(it)
        it=mk(tok,'icl_functask5p_copy',dd['copy'],qx,qx,alt=srt(qx),meta={'q':qx,'rule':'copy'}); it.twins['task']=it; out.append(it)
    return out

def fam_manyclass(tok,smoke,PER=None):
    """D63: activation-level binding capacity. K classes (K in 2,3,4,6) drawn from six sentence pools the early layers plausibly separate
    (sentiment+ / sentiment- / animals / food / Spanish / questions), ONE demo per class, arbitrary one-token labels; query = a held-out sentence of one class.
    Target = its class label; alt = the label of another class (random). top5/argmax logged -> K-way accuracy by argmax. Prediction (key-in-slot account):
    accuracy stays high while each class has its own slot and key direction; the per-item margin should fall roughly like log(K-1) only if keys are confusable."""
    T=json.load(open(_battery('icl_tasks.json')))['sentiment_sym']
    pools={'pos':[x for x,y in T if y=='foo'],'neg':[x for x,y in T if y=='bar'],'animal':CONCEPTS['topic'][0],'food':CONCEPTS['topic'][1],'spanish':CONCEPTS['lang'][1],'question':CONCEPTS['question'][0]}
    labels=['foo','bar','baz','vex','lux','rex']; out=[]; random.seed(63); NDEMO=int(os.environ.get('MANYCLASS_NDEMO','12'))
    for K in (2,3,4,6):
        for s_ in range(1 if smoke else 24):
            cls=random.sample(list(pools),K); labs=random.sample(labels,K); demos=[]; per=PER or NDEMO//K
            for c,lb in zip(cls,labs): demos+=[(x,lb) for x in random.sample(pools[c],per)]
            random.shuffle(demos); qi=random.randrange(K); qc=cls[qi]; used={x for x,_ in demos}; qs=random.choice([x for x in pools[qc] if x not in used]); alt=random.choice([l for l in labs if l!=labs[qi]])
            random.shuffle(demos)
            it=mk(tok,f'icl_manyclass_K{K}',demos,qs,labs[qi],alt=alt,meta={'q':qs,'K':K,'ndemo':per*K,'qclass':qc,'labels':dict(zip(cls,labs))}); it.twins['task']=it; out.append(it)
    return out

SG_SWAP=False
def fam_slotgatex(tok,smoke):
    """D84x: fam_slotgate with foo<->bar exchanged at every slot (target ' bar', alt ' foo'); same sentences, same seeds."""
    global SG_SWAP
    SG_SWAP=True
    try: return fam_slotgate(tok,smoke)
    finally: SG_SWAP=False

def fam_slotgate(tok,smoke):
    """D64: is the lone-slot failure on the query side? Length-matched twin pairs, K=2 classes A/B from the six pools, 4 demos:
    'one'  = A1->foo, B1->bar, A2->baz, B2->vex   (every label occurs once; A2/B2 re-labelled with fresh one-token labels)
    'two'  = A1->foo, B1->bar, A2->foo, B2->bar   (two slots per class)
    Same tokens except the two late label tokens, so the first two demos' K/V are IDENTICAL by construction (built-in null); query = a third A sentence,
    target foo, alt bar. Twin 'task' maps one<->two. Prediction: handing the fetch heads' Q at the query from 'two' into 'one' restores foo (Q-side gate)."""
    T=json.load(open(_battery('icl_tasks.json')))['sentiment_sym']
    pools={'pos':[x for x,y in T if y=='foo'],'neg':[x for x,y in T if y=='bar'],'animal':CONCEPTS['topic'][0],'food':CONCEPTS['topic'][1],'spanish':CONCEPTS['lang'][1],'question':CONCEPTS['question'][0]}
    out=[]; random.seed(64+int(os.environ.get('BATTERY_SEED','0'))); same=lambda t,it: len(t.ids)==len(it.ids) and [(p,q) for p,q,_ in t.spans]==[(p,q) for p,q,_ in it.spans]
    for s_ in range(1 if smoke else 30):
        A,B=random.sample(list(pools),2); a=random.sample(pools[A],3); b=random.sample(pools[B],2)
        first=[(a[0],'foo'),(b[0],'bar')]; random.shuffle(first)
        one=first+([(a[1],'baz'),(b[1],'vex')] if random.random()<0.5 else [(b[1],'vex'),(a[1],'baz')])
        two=[(x,{'baz':'foo','vex':'bar'}.get(y,y)) for x,y in one]
        tg,al=('bar','foo') if SG_SWAP else ('foo','bar')
        if SG_SWAP: one=[(x,{'foo':'bar','bar':'foo'}.get(y,y)) for x,y in one]; two=[(x,{'foo':'bar','bar':'foo'}.get(y,y)) for x,y in two]
        i1=mk(tok,'icl_slotgate_one',one,a[2],tg,alt=al,meta={'q':a[2],'rule':'one','A':A,'B':B}); i2=mk(tok,'icl_slotgate_two',two,a[2],tg,alt=al,meta={'q':a[2],'rule':'two','A':A,'B':B})
        if not same(i1,i2): continue
        i1.twins['task']=i2; i2.twins['task']=i1; out+=[i1,i2]
        nq=len(tok(a[2])['input_ids'])   # 'qx': query replaced by an unused B-class sentence of equal token length; target/alt exchanged (demos identical)
        for z in pools[B]:
            if z in b or len(tok(z)['input_ids'])!=nq: continue
            for it_,dd in ((i1,one),(i2,two)):
                t=mk(tok,it_.kind,dd,z,al,alt=tg,meta={**it_.meta,'qx':True})
                if same(t,it_): it_.twins['qx']=t
            if 'qx' in i2.twins and 'qx' in i1.twins: break
    return out

def fam_slotmix(tok,smoke):
    """D65: is the slot gate prompt-global (a query-side gain opened by ANY recurring label) or per-label (induction on that token)?
    Three classes A,B,C; kinds: 'c1' = A x3, B x3, C x1 (query C: its lone slot sits in a prompt where other labels recur);
    'c3' = A x3, B x3, C x3 (ceiling); 'c1all' = A x1, B x1, C x1 (floor, D63 regime). Target = C's label, alt = A's label; top5 gives argmax."""
    T=json.load(open(_battery('icl_tasks.json')))['sentiment_sym']
    pools={'pos':[x for x,y in T if y=='foo'],'neg':[x for x,y in T if y=='bar'],'animal':CONCEPTS['topic'][0],'food':CONCEPTS['topic'][1],'spanish':CONCEPTS['lang'][1],'question':CONCEPTS['question'][0]}
    out=[]; random.seed(65); labels=['foo','bar','baz','vex','lux','rex']
    for s_ in range(1 if smoke else 24):
        A,B,C=random.sample(list(pools),3); la,lb,lc=random.sample(labels,3)
        a=random.sample(pools[A],3); b=random.sample(pools[B],3); c=random.sample(pools[C],4)
        for kind,na,nb,nc in (('c1',3,3,1),('c3',3,3,3),('c1all',1,1,1)):
            demos=[(x,la) for x in a[:na]]+[(x,lb) for x in b[:nb]]+[(x,lc) for x in c[:nc]]; random.shuffle(demos)
            it=mk(tok,f'icl_slotmix_{kind}',demos,c[3],lc,alt=la,meta={'q':c[3],'rule':kind,'K':3,'qclass':C,'labels':{A:la,B:lb,C:lc}}); it.twins['task']=it; out.append(it)
    return out


def fam_mixed(tok,smoke):
    """D62: do the retrieval program and the function program coexist in ONE prompt? 8-shot prompts interleaving 4 sentiment demos (sentence -> good/bad; 2 pos + 2 neg)
    with 4 antonym demos (word -> antonym) in random order. Query kinds: 'sentq' (held-out sentence; answer = its label, alt = other label), 'wordq' (new word; answer = antonym,
    alt = the word) and 'wordq_copy' (same prompt with the 4 word demos as copy demos; answer = word, alt = antonym). 'task' twin = the word demos under the other rule (anto<->copy),
    sentiment demos identical: predicts a full rule flip for word queries and NO change for sentence queries. Pure controls: 'sentonly' (4 sentiment demos) and 'antoonly' (4 antonym demos)."""
    T=json.load(open(_battery('icl_tasks.json')))['sentiment_sym']; pos=[x for x,y in T if y=='foo']; neg=[x for x,y in T if y=='bar']; lp,ln='good','bad'
    out=[]; random.seed(62)
    same=lambda t,it: len(t.ids)==len(it.ids) and [(p,q) for p,q,_ in t.spans]==[(p,q) for p,q,_ in it.spans]
    for s_ in range(1 if smoke else 24):
        ps=random.sample(pos,3); ns=random.sample(neg,3); qpos=random.random()<0.5; qs=ps[2] if qpos else ns[2]; st=lp if qpos else ln; sa=ln if qpos else lp
        sd=[(x,lp) for x in ps[:2]]+[(x,ln) for x in ns[:2]]
        wp=random.sample(ANTONYMS,5); wd=wp[:4]; qx,qy=wp[4]; wc=[(x,x) for x,_ in wd]
        order=list(range(8)); random.shuffle(order)
        mixA=[(sd+wd)[i] for i in order]; mixC=[(sd+wc)[i] for i in order]
        sq=mk(tok,'icl_mixed_sentq',mixA,qs,st,alt=sa,meta={'q':qs,'rule':'anto','qpos':qpos}); tsq=mk(tok,'icl_mixed_sentq',mixC,qs,st,alt=sa,meta={'q':qs,'rule':'copy','qpos':qpos})
        if same(tsq,sq): sq.twins['task']=tsq
        # 'sentx' for the sentence query inside the MIXED prompt: exchange one same-class and one opposite-class sentiment sentence of equal token
        # length, labels and word demos in place (= labelswap sentx). Prediction: retrieval margin drops as in pure prompts (D61: +3.6 -> +0.9).
        sidx=[i for i,k in enumerate(order) if k<4]
        for i_ in sidx:
            for j_ in sidx:
                xi,yi=mixA[i_]; xj,yj=mixA[j_]
                if yi==yj or yi!=st or len(tok(xi)['input_ids'])!=len(tok(xj)['input_ids']): continue
                mx=list(mixA); mx[i_]=(xj,yi); mx[j_]=(xi,yj)
                tx=mk(tok,'icl_mixed_sentq',mx,qs,st,alt=sa,meta={'q':qs,'rule':'anto','qpos':qpos,'sentx':(i_,j_)})
                if same(tx,sq): sq.twins['sentx']=tx
                break
            if 'sentx' in sq.twins: break
        wq=mk(tok,'icl_mixed_wordq',mixA,qx,qy,alt=qx,meta={'q':qx,'rule':'anto'}); twq=mk(tok,'icl_mixed_wordq',mixC,qx,qy,alt=qx,meta={'q':qx,'rule':'anto'})
        if same(twq,wq): wq.twins['task']=twq
        cq=mk(tok,'icl_mixed_wordq_copy',mixC,qx,qx,alt=qy,meta={'q':qx,'rule':'copy'}); tcq=mk(tok,'icl_mixed_wordq_copy',mixA,qx,qx,alt=qy,meta={'q':qx,'rule':'copy'})
        if same(tcq,cq): cq.twins['task']=tcq
        so=mk(tok,'icl_mixed_sentonly',sd,qs,st,alt=sa,meta={'q':qs,'qpos':qpos}); so.twins['task']=so
        ao=mk(tok,'icl_mixed_antoonly',wd,qx,qy,alt=qx,meta={'q':qx}); ao.twins['task']=ao
        out+=[sq,wq,cq,so,ao]
    return out

FAMILIES={'slotgatex':fam_slotgatex,'concept12':fam_concept12,'labelswap':fam_labelswap,'mixed':fam_mixed,'func':fam_func,'func2':fam_func2,'functask':fam_functask,'functask2':fam_functask2,'functask3':fam_functask3,'functask4':fam_functask4,'functask5':fam_functask5, 'functask5p': fam_functask5p,'manyclass': fam_manyclass,'slotgate': fam_slotgate,'slotmix': fam_slotmix,'manyclass_per1': lambda t,s_: fam_manyclass(t,s_,PER=1), 'manyclass_per3': lambda t,s_: fam_manyclass(t,s_,3),'manyclass_per2': lambda t,s_: fam_manyclass(t,s_,2),'markcorrect':fam_markcorrect,'std':fam_std,'redund':fam_redund,'correct':fam_correct,'bind':fam_bind,'ctl':fam_ctl,'concept':fam_concept}
def build_battery(tok,families=('std','redund','correct','bind','ctl'),smoke=False):
    B=[]
    for f in families:
        rows=FAMILIES[f](tok,smoke); assert rows, f'family {f} produced no rows'; B+=rows
    kinds=sorted(set(b.kind for b in B)); print('BATTERY',len(B),'rows; kinds:',kinds,flush=True)
    return B
