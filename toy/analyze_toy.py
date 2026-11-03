"""Analysis loupes for the from-scratch toy (HF Llama). usage: python analyze_toy.py RUNDIR [--n 200] [--cpu]
Writes RUNDIR/analysis.json + prints: (1) query-label margin on unique vs recurring prompts; (2) class-flip twin DLA per head (readers = heads whose direct
contribution to the margin flips with the query class); (3) gate direction per layer = mean residual(arrow, recurring) - mean residual(arrow, unique) at the
query arrow, held-out sign; steering +/-g at the query arrow -> margin + readers' attention mass on bos vs matching slots; (4) read law: readers' matching
mass vs n_m for recurring prompts; (5) slot swap: exchange the two label tokens' residuals at layer l -> does the answer follow the slot."""
import sys,json,random,torch,math,collections,argparse,os
_argv=list(sys.argv); sys.argv=['x','--out','/tmp/_']; exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'train_toy.py')).read().split("if __name__")[0])
from transformers import LlamaForCausalLM
P=argparse.ArgumentParser(); P.add_argument('run'); P.add_argument('--n',type=int,default=200); P.add_argument('--cpu',action='store_true'); A=P.parse_args(_argv[1:])
dev='cpu' if A.cpu else 'cuda'; meta=json.load(open(f'{A.run}/train_meta.json')); cls_of={int(k):tuple(v) for k,v in meta['cls_of'].items()}
model=LlamaForCausalLM.from_pretrained(f'{A.run}/model',attn_implementation='eager').to(dev).eval(); L=model.config.num_hidden_layers; H=model.config.num_attention_heads; dh=model.config.hidden_size//H
r=random.Random(123)
def gen(kind,n,nm=None,K=None):
    out=[]
    while len(out)<n:
        seq,tgt,kd=prompt(r,'mix' if kind=='one' else 'noone',cls_of)
        if kd!=kind: continue
        out.append((seq[:-1],tgt[-1][1],seq))   # drop final label; target = query label
    return out
def margin(logits,tgt,seq):
    labs=sorted({t for t in seq if t>=L0}); alts=[l for l in labs if l!=tgt]; return float(logits[tgt]-max(logits[a] for a in alts)) if alts else float('nan')
REC={}; CAP={}
def cap_hook(l):
    def f(mod,inp): CAP[l]=inp[0].detach()
    return f
for l in range(L): model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(cap_hook(l))
ATT={}
def att_hook(l):
    def f(mod,inp,out):
        if out[1] is not None: ATT[l]=out[1].detach()
    return f
for l in range(L): model.model.layers[l].self_attn.register_forward_hook(att_hook(l))
RES={}; EDIT={}
def res_hook(l):
    def f(mod,inp,out):
        h=out[0] if isinstance(out,tuple) else out; RES[l]=h.detach()
        if l in EDIT: h=EDIT[l](h); return (h,)+tuple(out[1:]) if isinstance(out,tuple) else h
    return f
for l in range(L): model.model.layers[l].register_forward_hook(res_hook(l))
def run(seq):
    x=torch.tensor([seq],device=dev)
    with torch.no_grad(): out=model(input_ids=x,output_attentions=True)
    return out.logits[0,-1]
def dla(seq,tgt):   # per-head direct contribution to (tgt - best alt) through the final norm's scale at the last position
    lg=run(seq); labs=sorted({t for t in seq if t>=L0}); alt=max((l for l in labs if l!=tgt),key=lambda a:float(lg[a])); w=model.lm_head.weight[tgt]-model.lm_head.weight[alt]
    hN=RES[L-1][0,-1]; rho=hN.pow(2).mean().sqrt()+model.config.rms_norm_eps; g=model.model.norm.weight
    out={}
    for l in range(L):
        z=CAP[l][0,-1].view(H,dh); Wo=model.model.layers[l].self_attn.o_proj.weight.view(-1,H,dh)
        for h in range(H): out[f'{l}:{h}']=float((Wo[:,h,:]@z[h]*g/rho)@w)
    return out,float(lg[tgt]-lg[alt])
one=gen('one',A.n); two=gen('two',A.n)
REC['margin_one']=sum(margin(run(s),t,f) for s,t,f in one)/len(one); REC['margin_two']=sum(margin(run(s),t,f) for s,t,f in two)/len(two)
print('margin one %.2f two %.2f'%(REC['margin_one'],REC['margin_two']))
# (2) class-flip twin: replace the query sentence by a sentence of another demo class; DLA(intact) - DLA(flipped) per head
dl=collections.defaultdict(list)
for s,t,f in two[:A.n//2]:
    def cls(ws): return [c for c in range(K) if all(W0<=w<L0 and c in cls_of[w-W0] for w in ws)]
    arrows=[i for i,tok in enumerate(s) if tok==ARROW]; cl=sorted({c for i in arrows[:-1] for c in cls(s[i-3:i])}); qc=cls(s[-4:-1])[0]; others=[c for c in cl if c!=qc]
    if not others: continue
    s2=s[:-4]+[W0+w for w in sentence(r,r.choice(others),cls_of)]+[ARROW]; d1,m1=dla(s,t); d2,m2=dla(s2,t)
    for k in d1: dl[k].append(d1[k]-d2[k])
dlam={k:sum(v)/len(v) for k,v in dl.items()}; readers=sorted(dlam,key=lambda k:-dlam[k])[:4]; REC['dla_flip']=dlam; REC['readers']=readers
print('readers (class-flip DLA)',[(k,round(dlam[k],2)) for k in readers])
# (3) gate direction per layer at the query arrow
G={}; 
for l in range(L):
    a=torch.stack([ (run(s),RES[l][0,-1].clone())[1] for s,_,_ in one[:A.n//2]]); b=torch.stack([(run(s),RES[l][0,-1].clone())[1] for s,_,_ in two[:A.n//2]])
    g=(b.mean(0)-a.mean(0)); g=g/g.norm(); G[l]=g
    ho=[float(RES[l][0,-1]@g) for s,_,_ in one[A.n//2:] if run(s) is not None]; ht=[float(RES[l][0,-1]@g) for s,_,_ in two[A.n//2:] if run(s) is not None]
    thr=(sum(ho)/len(ho)+sum(ht)/len(ht))/2; acc=(sum(x<thr for x in ho)+sum(x>thr for x in ht))/(len(ho)+len(ht)); REC[f'gate_L{l}']=dict(gap=sum(ht)/len(ht)-sum(ho)/len(ho),heldout_acc=acc)
    print(f'gate L{l}: gap {REC[f"gate_L{l}"]["gap"]:.2f} held-out sign acc {acc:.2f}')
def steer(kind_set,l,scale,rand=False):
    gvec=G[l]*scale*abs(REC[f'gate_L{l}']['gap'])   # dose in units of the natural recurring-minus-unique gap along g at this layer (scale -1 moves a recurring prompt's projection to the unique mean)
    if rand: _g=torch.Generator().manual_seed(777+l); _v=torch.randn(G[l].shape,generator=_g).to(G[l].device); gvec=_v/_v.norm()*gvec.norm()   # random direction, same norm: control for the steer
    def ed(h): h=h.clone(); h[0,-1]+=gvec; return h
    EDIT[l]=ed; ms=[];sink=[];mm=[]
    for s,t,f in kind_set:
        lg=run(s); ms.append(margin(lg,t,f))
        slots=[i for i,tok in enumerate(s) if i>0 and s[i-1]==ARROW and tok>=L0]; match=[i for i in slots if s[i]==t]
        for rk in readers[:3]:
            ll,hh=map(int,rk.split(':')); a=ATT[ll][0,hh,-1]; sink.append(float(a[0])); mm.append(float(a[match].sum()) if match else 0.0)
    EDIT.pop(l,None); return dict(margin=sum(ms)/len(ms),sink=sum(sink)/len(sink),match=sum(mm)/len(mm))
gl=max(range(L-1),key=lambda l:REC[f'gate_L{l}']['gap']); REC['gate_layer']=gl
for kind,ks in (('one',one),('two',two)):
    for sc in (-2,-1,0,1,2): REC[f'steer_{kind}_{sc:+d}']=steer(ks,gl,sc); print(f'steer {kind} {sc:+d}@L{gl}',{k:round(v,2) for k,v in REC[f'steer_{kind}_{sc:+d}'].items()})
    for sc in (-2,-1,1,2): REC[f'steerrand_{kind}_{sc:+d}']=steer(ks,gl,sc,rand=True); print(f'steer-rand {kind} {sc:+d}@L{gl}',{k:round(v,2) for k,v in REC[f'steerrand_{kind}_{sc:+d}'].items()})
# (4) read law: readers' matching mass vs n_m (recurring prompts)
law=collections.defaultdict(list)
for s,t,f in two:
    run(s); slots=[i for i,tok in enumerate(s) if i>0 and s[i-1]==ARROW and tok>=L0]; match=[i for i in slots if s[i]==t]
    for rk in readers[:3]:
        ll,hh=map(int,rk.split(':')); a=ATT[ll][0,hh,-1]; law[(rk,len(match),len(slots))].append((float(a[match].sum()),float(a[0])))
REC['law']={f'{k[0]}|nm{k[1]}|N{k[2]}':dict(n=len(v),m=sum(x for x,_ in v)/len(v),sink=sum(y for _,y in v)/len(v)) for k,v in law.items() if len(v)>=5}
# (5) slot swap at layer l: exchange the residuals of the matching slot and a non-matching slot -> does the answer follow?
sw={}
for l in range(L-1):
    foll=[]
    for s,t,f in two[:A.n//2]:
        slots=[i for i,tok in enumerate(s) if i>0 and s[i-1]==ARROW and tok>=L0]; match=[i for i in slots if s[i]==t]; non=[i for i in slots if s[i]!=t]
        if not match or not non: continue
        a,b=match[0],non[0]
        def ed(h,a=a,b=b): h=h.clone(); h[0,[a,b]]=h[0,[b,a]]; return h
        EDIT[l]=ed; lg=run(s); EDIT.pop(l,None); foll.append(float(lg[s[b]]-lg[t]))
    sw[l]=sum(foll)/len(foll); print(f'slot swap @L{l}: (swapped-in label - true label) logit {sw[l]:+.2f}')
REC['slot_swap']=sw
json.dump(REC,open(f'{A.run}/analysis.json','w'),indent=1,default=float); print('saved')
