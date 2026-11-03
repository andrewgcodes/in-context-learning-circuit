"""Pure scoring helpers on tensors (unit-tested on synthetic inputs)."""
import torch,torch.nn.functional as F
def seq_logprob(lp,T,tgt):
    """lp: [T+len(tgt), V] log-softmax of prompt+answer; T: prompt length. Returns (sum log P(answer tokens), log P(first token))."""
    pos=torch.arange(T-1,T-1+len(tgt)); tgt=torch.as_tensor(tgt,device=lp.device)
    return float(lp[pos,tgt].sum()),float(lp[T-1,tgt[0]])
def surprisal(lp,ids):
    T=len(ids); return (-lp[torch.arange(T-1),torch.as_tensor(ids[1:],device=lp.device)]).cpu()
def flag_readout(kin,U,MU,SD,T):
    """Per-token projection of the late-layer key-input onto the flag direction U[l], in natural-text SD units."""
    return {str(l):[round(x,3) for x in (((kin[l][:T].float()@U[l]).cpu()-MU[l])/SD[l]).tolist()] for l in U}
def query_attention(att_by_layer,T,spans,tol=1e-3):
    """att_by_layer: {l: [heads,T,T]}; mean over heads of query-row attention mass on each label span. tol: row-sum tolerance (1e-3 fp32; bf16 attention rows sum to 1 only within ~1e-2)."""
    for l,a in list(att_by_layer.items())[:1]:
        row=a[:,T-1,:T].float().sum(-1); assert float((row-1).abs().max())<tol, ('attention row must sum to 1 per head',tol,row.tolist())
    return {str(l):[round(float(a[:,T-1,s0:s1].sum(-1).mean()),5) for (s0,s1,*_) in spans] for l,a in att_by_layer.items()}
def query_attention_heads(att_by_layer,T,spans,dup):
    """per-head query-row attention mass summed over label spans that are repeats (span start in dup) and over fresh ones: {l: {'rep':[heads], 'fresh':[heads]}}; a group with no spans gives None."""
    rep=[(s0,s1) for (s0,s1,*_) in spans if s0 in dup]; fresh=[(s0,s1) for (s0,s1,*_) in spans if s0 not in dup]
    def mass(a,grp): return [round(float(x),5) for x in sum(a[:,T-1,s0:s1].sum(-1) for (s0,s1) in grp).tolist()] if grp else None
    return {str(l):{'rep':mass(a,rep),'fresh':mass(a,fresh)} for l,a in att_by_layer.items()}
def dup_head_scores(att,ids):
    """att: [heads,T,T] attention for one layer; ids: list[int]. For every duplicate token t (a token that occurred earlier),
    look-back score = attention mass on its earlier copies, induction score = mass on the tokens right after those copies (j+1<t).
    Returns (lookback[heads], induction[heads]) averaged over duplicate tokens, or None if there are none. Vectorised: builds the two masks once."""
    T=len(ids); x=torch.as_tensor(ids,device=att.device); same=(x[:,None]==x[None,:]).tril(-1)   # same[t,j]: j<t and ids[j]==ids[t]
    dupq=same.any(-1)
    if not bool(dupq.any()): return None
    nxt=torch.zeros_like(same); nxt[:,1:]=same[:,:-1]; nxt=nxt.tril(-1)                          # nxt[t,j+1]=same[t,j], only j+1<t
    A=att.float()
    lb=(A*same[None].float()).sum(-1)[:,dupq].mean(-1); ind=(A*nxt[None].float()).sum(-1)[:,dupq].mean(-1)
    return lb.cpu(),ind.cpu()

def query_row_top(att_by_layer,T,heads,ids,tok,k=8):
    """Full attention-row loupe for named heads: at the query row (position T-1) list the k most-attended positions with their
    index, decoded token and weight, plus the total mass on them. heads: {layer: [head,...]} ('all' -> every head).
    Returns {f'{l}:{h}': {'top':[[pos,token,weight],...], 'mass':float}}. Layers absent from att_by_layer are skipped."""
    out={}
    for l,hs in heads.items():
        if l not in att_by_layer: continue
        a=att_by_layer[l]; hs=range(a.shape[0]) if hs=='all' else hs
        for h in hs:
            row=a[h,T-1,:T].float(); w,p=row.topk(min(k,T))
            out[f'{l}:{h}']={'top':[[int(pi),tok.decode([int(ids[pi])]),round(float(wi),5)] for wi,pi in zip(w.tolist(),p.tolist())],'mass':round(float(w.sum()),5)}
    return out


def mlp_direct(h,W,d,k=5):
    """Direct push of one MLP layer's output onto direction d at one position. h: [ffn] post-activation hidden (input to the down
    projection), W: [d_model, ffn] down matrix, d: [d_model] (target minus alternative unembedding). Returns (total, top-k neurons by
    |contribution| as [[neuron, contribution], ...]); total == (W @ h) . d exactly."""
    c=h.float()*(W.float().T@d.float()); top=c.abs().topk(min(k,c.numel())).indices
    return float(c.sum()),[[int(i),round(float(c[i]),3)] for i in top.tolist()]

def slot_rows(att_by_layer,spans,ids,tok,layers):
    """Slot-position attention loupe: for every head in `layers`, the attention row FROM each demo label slot (first label token of each span),
    averaged over slots: mass on the slot's own sentence tokens (after the previous label, before its arrow), on its arrow, on other demos' sentence
    tokens, on other label tokens, on position 0 (sink), on itself; plus the most-attended own-sentence token (decoded) and its weight, and how often
    the same own-sentence offset-from-arrow wins across slots. Returns {'l:h': [own,arrow,othersent,otherlab,sink,self,top_tok,top_w,top_share]}."""
    out={}; sp=sorted(spans,key=lambda s:s[0]); labs=set(); arrows=set(); own={}
    for i,s in enumerate(sp):
        s0,s1=s[0],s[1]; labs|=set(range(s0,s1)); arrows.add(s0-1)
        lo=sp[i-1][1] if i else 0; own[s0]=[p for p in range(lo,s0-1) if tok.decode([int(ids[p])]).strip() not in ('','->','-','>')]
        arrows|=set(range(lo,s0-1))-set(own[s0])
    for l in layers:
        if l not in att_by_layer: continue
        a=att_by_layer[l].float()
        for h in range(a.shape[0]):
            acc=[0.0]*6; tops=[]
            for s in sp:
                s0=s[0]; row=a[h,s0]; o=own[s0]
                if not o: continue
                ow=row[o]; j=int(ow.argmax()); tops.append((o[j],float(ow[j]),s0-1-o[j]))
                acc[0]+=float(ow.sum()); acc[1]+=float(sum(row[p] for p in range(own[s0][0] if o else s0-1,s0) if p not in o))
                acc[2]+=float(sum(row[p] for p in range(s0) if p not in o and p not in labs and p not in arrows and p!=0))
                acc[3]+=float(sum(row[p] for p in labs if p<s0)); acc[4]+=float(row[0]); acc[5]+=float(row[s0])
            n=len(tops)
            if not n: continue
            best=max(tops,key=lambda t:t[1]); offs=[t[2] for t in tops]; share=max(offs.count(x) for x in set(offs))/n
            out[f'{l}:{h}']=[round(x/n,4) for x in acc]+[tok.decode([int(ids[best[0]])]),round(best[1],4),round(share,2)]
    return out
