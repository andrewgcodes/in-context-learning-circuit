"""D172 build-from-parts B: train a tiny HF-Llama from scratch on a synthetic slotgate distribution.
Tokens: 0 pad,1 bos,2 arrow,3 newline, 4..4+W-1 words (W=48, each word belongs to 2 of K=8 classes; a sentence = 3 words whose class sets intersect in exactly one class),
4+W.. labels (32). Prompt = bos, then demos 'w w w -> label \n', then query 'w w w ->'; loss on every label token after an arrow.
cond 'mix': 50% recurring prompts (each class 1-3 demos, consistent arbitrary label per class), 50% unique prompts (every demo a fresh class with a fresh label; the query class
is among the demos). cond 'gate': 50% recurring, 50% noise-unique (unique demo labels, fresh query label: retrieval is wrong, so label recurrence is the only certificate of a consistent mapping -> a recurrence gate must form). cond 'noone': recurring only (no unique-label prompts -> prediction: no recurrence gate forms). Labels redrawn per prompt.
usage: python train_toy.py --cond mix --seed 0 --steps 6000 --out DIR [--cpu]"""
import argparse,json,random,itertools,os,time,torch
from transformers import LlamaConfig,LlamaForCausalLM
P=argparse.ArgumentParser(); P.add_argument('--cond',default='mix'); P.add_argument('--seed',type=int,default=0); P.add_argument('--steps',type=int,default=6000); P.add_argument('--out',required=True); P.add_argument('--bs',type=int,default=64); P.add_argument('--layers',type=int,default=4); P.add_argument('--cpu',action='store_true')
A=P.parse_args(); K,W,NL=8,48,32; PAD,BOS,ARROW,NL_TOK=0,1,2,3; W0=4; L0=W0+W; VOCAB=L0+NL
def make_words(seed):
    r=random.Random(1000+seed); pairs=list(itertools.combinations(range(K),2)); r.shuffle(pairs)  # 28 pairs
    cls_of={w:pairs[w%len(pairs)] for w in range(W)}  # each word in 2 classes; 48 words cover each pair ~1.7x
    return cls_of
def sentence(r,c,cls_of):
    cand=[w for w in range(W) if c in cls_of[w]]
    while True:
        ws=r.sample(cand,3); common=set(range(K))
        for w in ws: common&=set(cls_of[w])
        if common=={c}: return ws
def prompt(r,cond,cls_of):
    kind='one' if (cond=='mix' and r.random()<0.5) else ('noise' if (cond=='gate' and r.random()<0.5) else 'two')
    if kind=='two':
        classes=r.sample(range(K),r.randint(2,4)); labels=dict(zip(classes,r.sample(range(NL),len(classes)))); demos=[c for c in classes for _ in range(r.randint(1,3))]; r.shuffle(demos); q=r.choice(classes)
        seq=[BOS]; tgt=[]
        for c in demos: seq+=[W0+w for w in sentence(r,c,cls_of)]+[ARROW]; tgt.append((len(seq),L0+labels[c])); seq+=[L0+labels[c],NL_TOK]
        seq+=[W0+w for w in sentence(r,q,cls_of)]+[ARROW]; tgt.append((len(seq),L0+labels[q])); seq+=[L0+labels[q]]
    else:  # unique: every demo a different class and a different label; query = one of the demo classes -> its label is predictable from one slot
        classes=r.sample(range(K),r.randint(3,6)); labs=r.sample(range(NL),len(classes)); q=r.choice(range(len(classes))); seq=[BOS]; tgt=[]
        for c,l in zip(classes,labs): seq+=[W0+w for w in sentence(r,c,cls_of)]+[ARROW]; tgt.append((len(seq),L0+l)); seq+=[L0+l,NL_TOK]
        ql=labs[q] if kind=='one' else r.choice([l for l in range(NL) if l not in labs])   # noise: unique demo labels do not certify a consistent mapping; the query label is fresh, so retrieving a slot is wrong
        seq+=[W0+w for w in sentence(r,classes[q],cls_of)]+[ARROW]; tgt.append((len(seq),L0+ql)); seq+=[L0+ql]
    return seq,tgt,kind
def batch(r,cond,cls_of,bs,T=96):
    X=torch.full((bs,T),PAD); Y=torch.full((bs,T),-100); kinds=[]
    for i in range(bs):
        seq,tgt,kind=prompt(r,cond,cls_of); seq=seq[:T]; X[i,:len(seq)]=torch.tensor(seq); kinds.append(kind)
        for pos,lab in tgt:
            if pos-1<T-0 and pos<=T-0 and pos-1>=0 and pos<T+1: Y[i,pos-1]=lab   # predict label at the arrow position (next token)
    return X,Y,kinds
if __name__=='__main__':
    torch.manual_seed(A.seed); r=random.Random(A.seed); cls_of=make_words(A.seed); os.makedirs(A.out,exist_ok=True); dev='cpu' if A.cpu else 'cuda'
    cfg=LlamaConfig(vocab_size=VOCAB,hidden_size=256,intermediate_size=512,num_hidden_layers=A.layers,num_attention_heads=8,num_key_value_heads=8,max_position_embeddings=128,rms_norm_eps=1e-6,tie_word_embeddings=False,bos_token_id=BOS,eos_token_id=NL_TOK,pad_token_id=PAD)
    model=LlamaForCausalLM(cfg).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.1,betas=(0.9,0.95)); sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min(1,(s+1)/200)*max(0.05,1-s/A.steps))
    json.dump(dict(cond=A.cond,seed=A.seed,steps=A.steps,K=K,W=W,NL=NL,cls_of={str(k):v for k,v in cls_of.items()},params=sum(p.numel() for p in model.parameters())),open(f'{A.out}/train_meta.json','w'))
    log=open(f'{A.out}/train_log.jsonl','a'); t0=time.time()
    for step in range(A.steps):
        X,Y,kinds=batch(r,A.cond,cls_of,A.bs); X,Y=X.to(dev),Y.to(dev); out=model(input_ids=X,labels=None); lg=out.logits[:,:-1]; yy=Y[:,:-1]
        loss=torch.nn.functional.cross_entropy(lg.reshape(-1,VOCAB),yy.reshape(-1),ignore_index=-100); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step()
        if step%50==0 or step==A.steps-1:
            with torch.no_grad():  # loss on the QUERY label only, split by prompt kind
                qloss={}
                for kd in ('one','two','noise'):
                    idx=[i for i,k in enumerate(kinds) if k==kd]
                    if not idx: continue
                    last=[(Y[i]!=-100).nonzero()[-1].item() for i in idx]; qloss[kd]=float(torch.nn.functional.cross_entropy(torch.stack([out.logits[i,p] for i,p in zip(idx,last)]),torch.stack([Y[i,p] for i,p in zip(idx,last)])))
            rec=dict(step=step,loss=float(loss),qloss=qloss,lr=sched.get_last_lr()[0],t=round(time.time()-t0,1)); log.write(json.dumps(rec)+'\n'); log.flush(); print(rec,flush=True)
    model.save_pretrained(f'{A.out}/model'); json.dump(dict(done=True,steps=A.steps),open(f'{A.out}/run.status','w')); print('saved',A.out)
