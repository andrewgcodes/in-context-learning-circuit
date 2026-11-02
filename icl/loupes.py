"""Per-token instruments beyond the basic readout (pure tensor functions; wired into run.py after the acceptance diff).
mlp_neuron_writers: which MLP neurons at layer l write onto direction u at the given positions (needs the MLP act [T,ffn] and down-proj weight [d,ffn]).
attn_logit_split: for a reader head, split the pre-softmax logit q.k at (query, key j) into the part along the flag direction and the rest.
value_readout: what a head moves when it attends to key position j: v_j through the head's OV slice, projected on a direction (e.g. answer unembedding)."""
import torch,torch.nn.functional as F
def mlp_neuron_writers(act,Wdown,u,positions,k=10):
    """act: [T,ffn] post-activation MLP hidden; Wdown: [d,ffn]; u: [d] unit. Returns per position the top-k neurons by signed write onto u and the total write."""
    wu=(u.to(Wdown.device)@Wdown)            # [ffn] write of each neuron's unit activation onto u
    out={}
    for p in positions:
        c=act[p].to(wu.device)*wu            # [ffn] contribution per neuron
        tot=float(c.sum()); top=torch.topk(c.abs(),min(k,c.numel()))
        out[int(p)]={'total':round(tot,4),'top':[(int(i),round(float(c[i]),4)) for i in top.indices]}
    return out
def attn_logit_split(q,K,u,scale):
    """q: [hd] post-norm query for one head at the query position; K: [T,hd] post-norm keys for that head; u: [hd] unit flag direction in key space.
    Returns (logit_total[T], logit_along_u[T], logit_rest[T]) with logit = scale*q.k."""
    u=u.to(K.device)/u.norm(); Ku=(K@u)[:,None]*u[None]; Kr=K-Ku
    return scale*(K@q),scale*(Ku@q),scale*(Kr@q)
def value_readout(V,Wo_h,direction):
    """V: [T,hd] values for one head; Wo_h: [hd,d] this head's slice of the output projection; direction: [d]. Returns [T] how much attending fully to key j moves the residual along direction."""
    return (V@Wo_h)@direction.to(V.device)
