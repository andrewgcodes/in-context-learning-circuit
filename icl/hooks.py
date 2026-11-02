"""Model-generic hooks: capture k_proj inputs (kin), post-norm q/k (qn,kn), attention-output-projection inputs (oin), GatedDeltaNet outputs (GOUT);
head mean-ablation knockouts (KNOCK: layer->heads, mean from MEANO) and GDN-layer knockouts (GKNOCK with MEANG). Only this file knows Qwen/GPT-2/hybrid layout."""
import torch
def attmod(lay): return getattr(lay,'self_attn',None) or getattr(lay,'attention',None) or getattr(lay,'attn',None)
def kstyle(lay):
    a=attmod(lay)
    if a is None: return None
    if hasattr(a,'k_proj'): return 'kproj'
    if hasattr(a,'c_attn'): return 'gpt2'
    if hasattr(a,'query_key_value'): return 'neox'   # GPT-NeoX/Pythia: fused QKV, per-head [q|k|v] blocks
    return None
def out_proj(lay):
    """The attention output projection module (its input is the concatenated per-head outputs)."""
    a=attmod(lay); st=kstyle(lay); return {'kproj':getattr(a,'o_proj',None),'gpt2':getattr(a,'c_proj',None),'neox':getattr(a,'dense',None)}[st]
def out_proj_matrix(lay):
    """[nh*hd, d] matrix mapping concatenated head outputs into the residual stream."""
    w=out_proj(lay).weight.float(); return w if kstyle(lay)=='gpt2' else w.T    # Conv1D stores [in,out]; Linear stores [out,in]
def attn_by_layer(o,SOFT,L):
    """Map layer index -> [heads,T,T] attention from a model output, for both tuple layouts (one entry per layer with None for non-softmax layers, or one entry per softmax layer)."""
    A=o.attentions
    if len(A)==L: return {l:a[0] for l,a in enumerate(A) if a is not None}
    if len(A)==len(SOFT): return {l:a[0] for l,a in zip(SOFT,A) if a is not None}
    raise ValueError(f'attention tuple length {len(A)} matches neither L={L} nor softmax layers={len(SOFT)}')
def get_layers(m):
    for path in ('model.layers','transformer.h','model.decoder.layers','gpt_neox.layers'):
        o=m
        try:
            for p in path.split('.'): o=getattr(o,p)
            return list(o)
        except AttributeError: pass
    raise ValueError('no layer list found')
class Hooks:
    def __init__(self,m,late):
        self.m=m; self.layers=get_layers(m); L=len(self.layers); c=m.config
        self.nh=c.num_attention_heads; self.hd=getattr(c,'head_dim',None) or c.hidden_size//self.nh; self.nkv=getattr(c,'num_key_value_heads',self.nh); self.Hs=c.hidden_size
        self.SOFT=[l for l in range(L) if kstyle(self.layers[l])]; self.GDN=[l for l in range(L) if getattr(self.layers[l],'linear_attn',None) is not None]
        self.late=late; self.kin={}; self.qn={}; self.kn={}; self.oin={}; self.GOUT={}; self.KDIRS={}; self.PATCH={}; self.PATCH_N=None; self.PATCH_POS=None; self.KNOCK_POS=None; self.KNOCK_ZERO=False; self.min={}; self.fin=None; self.REM={}; self.AOUT={}
        self.KNOCK={}; self.GKNOCK=set(); self.MEANO={}; self.MEANG={}; self.handles=[]; self.mlpin={}; self.vin={}; self.gate={}; self.up={}
        for l in late:
            mlp=getattr(self.layers[l],'mlp',None); down=getattr(mlp,'down_proj',None) or getattr(mlp,'c_proj',None) or getattr(mlp,'dense_4h_to_h',None)
        self.NEUR={}; self.DIRS={}
        for l in range(len(self.layers)):
            mlpm=getattr(self.layers[l],'mlp',None); om=out_proj(self.layers[l]) if attmod(self.layers[l]) is not None else None
            if mlpm is not None: self.handles.append(mlpm.register_forward_hook(self._dir_post(l,'mlp')))
            if om is not None: self.handles.append(om.register_forward_hook(self._dir_post(l,'attn')))
            mlp=getattr(self.layers[l],'mlp',None); down=getattr(mlp,'down_proj',None) or getattr(mlp,'c_proj',None) or getattr(mlp,'dense_4h_to_h',None)
            if down is not None: self.handles.append(down.register_forward_pre_hook(self._neur_pre(l)))
            if l in late:
                for nm in ('gate_proj','up_proj'):
                    pm=getattr(mlp,nm,None)
                    if pm is not None: self.handles.append(pm.register_forward_hook(self._swiglu_post(l,nm)))
            self.down={l:(getattr(mlp,'down_proj',None) or getattr(mlp,'c_proj',None) or getattr(mlp,'dense_4h_to_h',None)) for l in late}
        nh,hd,nkv,Hs=self.nh,self.hd,self.nkv,self.Hs
        fn=getattr(getattr(m,'model',None),'norm',None) or getattr(getattr(m,'transformer',None),'ln_f',None) or getattr(getattr(m,'gpt_neox',None),'final_layer_norm',None)
        if fn is not None: self.handles.append(fn.register_forward_pre_hook(self._final_pre))
        self.RES={}; self.RESPOS=None; self.DADD={}   # residual-stream loupe: RES[l]=layer-l input at RESPOS; DADD[l]=(vec[d], positions, scale) added to the layer-l input
        for l in range(len(self.layers)):
            self.handles.append(self.layers[l].register_forward_pre_hook(self._layer_pre(l),with_kwargs=True))
            if getattr(self.layers[l],'mlp',None) is not None: self.handles.append(self.layers[l].mlp.register_forward_pre_hook(self._mlp_pre(l)))
        for l in self.SOFT:
            a=attmod(self.layers[l]); st=kstyle(self.layers[l])
            if st=='kproj':
                self.handles.append(a.k_proj.register_forward_pre_hook(lambda mod,args,l=l: self.kin.__setitem__(l,args[0][0].detach().clone())))
                self.handles.append(a.k_proj.register_forward_hook(self._kdir_post(l,'k'))); self.handles.append(a.q_proj.register_forward_hook(self._kdir_post(l,'q'))); self.handles.append(a.v_proj.register_forward_hook(self._kdir_post(l,'v')))
                if l in late:
                    if hasattr(a,'q_norm'):
                        self.handles.append(a.q_norm.register_forward_hook(lambda mod,args,out,l=l: self.qn.__setitem__(l,out[0].detach().float().clone())))
                        self.handles.append(a.k_norm.register_forward_hook(lambda mod,args,out,l=l: self.kn.__setitem__(l,out[0].detach().float().clone())))
                proj=a.o_proj
            elif st=='gpt2':
                if l in late:
                    def post(mod,args,out,l=l):
                        y=out[0] if isinstance(out,tuple) else out
                        self.kin[l]=y[0,:,Hs:2*Hs].detach().clone(); self.qn[l]=y[0,:,:Hs].detach().float().view(-1,nh,hd).clone(); self.kn[l]=y[0,:,Hs:2*Hs].detach().float().view(-1,nkv,hd).clone(); self.vin[l]=y[0,:,2*Hs:3*Hs].detach().float().view(-1,nkv,hd).clone()
                    self.handles.append(a.c_attn.register_forward_hook(post))
                proj=a.c_proj
            else:   # neox: qkv [T, nh, 3*hd] per head -> q=[:hd], k=[hd:2hd], v=[2hd:]
                if l in late:
                    def post(mod,args,out,l=l):
                        y=(out[0] if isinstance(out,tuple) else out)[0].view(-1,nh,3*hd)
                        self.kin[l]=y[:,:,hd:2*hd].reshape(-1,nh*hd).detach().clone(); self.qn[l]=y[:,:,:hd].detach().float().clone(); self.kn[l]=y[:,:,hd:2*hd].detach().float().clone(); self.vin[l]=y[:,:,2*hd:].detach().float().clone()
                    self.handles.append(a.query_key_value.register_forward_hook(post))
                proj=a.dense
            def pre(mod,args,l=l):
                x=args[0]; self.oin[l]=x[0].detach().float().clone()
                hp=self.PATCH.get(l,{}).get('hout')
                if l not in self.KNOCK and hp is None and not (self.PATCH.get(l,{}).get('houtu') is not None or self.PATCH.get(l,{}).get('houtp') is not None): return None
                x=x.clone(); hdo=x.shape[-1]//nh; mu=self.MEANO.get(l)
                if hp is not None:   # per-head attention output (o_proj input slice) handed from the twin at PATCH_POS
                    s_,hs_=hp
                    for h in hs_: x[...,h*hdo:(h+1)*hdo]=self._blend(s_[:,h*hdo:(h+1)*hdo],x[...,h*hdo:(h+1)*hdo])
                    self.oin[l]=x[0].detach().float().clone()
                for w_ in ('houtu','houtp'):
                    hp2=self.PATCH.get(l,{}).get(w_)
                    if hp2 is None: continue
                    s_,hs_,u_=hp2; Wo=mod.weight.float(); B_=(u_.float() if u_.dim()==2 else (u_.float()/u_.float().norm())[:,None]).to(Wo.device)   # [d,m] orthonormal columns
                    for h in hs_:
                        Woh=Wo[:,h*hdo:(h+1)*hdo]; own=x[...,h*hdo:(h+1)*hdo].float(); tw=self._blend(s_[:,h*hdo:(h+1)*hdo].float(),own.clone())
                        dr=(tw-own)@Woh.T                      # twin-minus-own residual write of head h at every position (zero outside PATCH_POS by _blend)
                        du=(dr@B_)@B_.T                         # its component inside the class subspace
                        keep=du if w_=='houtu' else dr-du
                        x[...,h*hdo:(h+1)*hdo]=(own+keep@torch.linalg.pinv(Woh).T).to(x.dtype)   # least-squares pre-image in o_proj input space
                    self.oin[l]=x[0].detach().float().clone()
                for h in self.KNOCK.get(l,()):
                    rep=(mu[h*hdo:(h+1)*hdo].to(x.dtype) if (mu is not None and not self.KNOCK_ZERO) else 0)
                    if self.KNOCK_POS is None: x[...,h*hdo:(h+1)*hdo]=rep
                    else: x[:,self.KNOCK_POS,h*hdo:(h+1)*hdo]=rep   # positions only (e.g. [-1] = query)
                return (x,)+tuple(args[1:])
            self.handles.append(proj.register_forward_pre_hook(pre))
        for l in self.GDN:
            def gpost(mod,args,out,l=l):
                y=out[0] if isinstance(out,tuple) else out; self.GOUT[l]=y[0].detach().float().clone()
                if l in self.GKNOCK and l in self.MEANG:
                    y2=self.MEANG[l].to(y.dtype).expand_as(y[0])[None]; return (y2,)+tuple(out[1:]) if isinstance(out,tuple) else y2
                return None
            self.handles.append(self.layers[l].linear_attn.register_forward_hook(gpost))
    def set_neurons(self,spec):
        """spec: list of (layer, neuron_idx) to zero at the MLP down-projection input (acts after the gate/activation); None clears."""
        self.NEUR={}
        for l,i in (spec or []): self.NEUR.setdefault(l,[]).append(i)
    def _neur_pre(self,l):
        # zero the named neurons at the down-projection input, then record what the down projection actually sees (late layers) so the
        # MLP direct-logit / neuron loupes read the intervened state, never the pre-intervention one
        def pre(mod,args):
            idx=self.NEUR.get(l); x=args[0]
            if idx: x=x.clone(); x[...,idx]=0
            if l in self.late or getattr(self,'MLPIN_ALL',False): self.mlpin[l]=x[0].detach().float().clone()
            return (x,)+tuple(args[1:]) if idx else None
        return pre
    def _swiglu_post(self,l,nm):
        # SwiGLU loupe: record the gate pre-activation (gate_proj output, before silu) and the up value at late layers, so a neuron's
        # change under an intervention can be split into gate-side (switching) vs up-side (value) — read with H.gate[l], H.up[l]
        def post(mod,args,out):
            (self.gate if nm=='gate_proj' else self.up)[l]=out[0].detach().float().clone()
        return post
    def set_dirs(self,spec):
        """spec: {layer: {'mlp': unit vec (d_model) | 'attn': unit vec}} — project the given residual-space direction out of that layer's MLP output / attention o_proj output at every position. None clears."""
        self.DIRS=spec or {}
    def set_kdirs(self,spec):
        """spec: {layer: {'k'|'v': (u, [kv groups]) | 'q': (u, [heads])}} (several keys per layer allowed) with u a unit vector in the k_proj-INPUT space of that layer (the space U[l] lives in).
        Path-specific flag removal: the selected KV groups' keys (or heads' queries) are computed from x - (x.u)u while every other reader of x sees x unchanged."""
        self.KDIRS=spec or {}
    def _kdir_post(self,l,which):
        def post(mod,args,out):
            ps=self.PATCH.get(l,{}).get(which)
            if ps is not None:
                xs,idx=ps; hd=self.hd; ys=torch.nn.functional.linear(xs.to(out.device,out.dtype)[None],mod.weight,mod.bias)   # source projection, every position
                y=out.clone(); sl=torch.cat([torch.arange(g*hd,(g+1)*hd) for g in idx]).to(out.device)
                return self._blend(ys[0],y,fn=lambda d,s:d.index_copy(-1,sl,s[...,sl]))
            sp=self.KDIRS.get(l,{}).get(which)
            if sp is None: return None
            u,idx=sp; x=args[0]; u=u.to(x.device,x.dtype); hd=self.hd
            c=(x@u)[...,None]; wu=(mod.weight@u)            # (x.u) and W u  -> out slice -= (x.u) * (W u)_slice
            y=out.clone()
            for g in idx: y[...,g*hd:(g+1)*hd]=out[...,g*hd:(g+1)*hd]-c*wu[g*hd:(g+1)*hd]
            return y
        return post
    def set_patch(self,spec):
        """Activation patching of reader INPUTS from a source run. spec: {layer: {'k'|'v': (x_src[T,d], [kv groups]), 'q': (x_src, [heads]), 'mlp': x_src}, 'final': x_src, 'final_delta': d[T,d] added to the final-norm input (direct-path patch: -sum of removed writes)}
        x_src = post-norm input captured from the source (e.g. flag-removed) forward (self.kin[l] / self.min[l] / self.fin). Only the named readers see the source input."""
        self.PATCH=spec or {}
    _FWD=('kin','qn','kn','oin','min','fin','REM','AOUT','mlpin','vin','GOUT')
    def stash(self):
        """Set aside the current forward's recorded state so another forward (e.g. the alternative-answer tail) can run without
        overwriting it; pass the result to unstash() afterwards so every loupe reads the target forward."""
        st={k:getattr(self,k) for k in self._FWD}
        for k in self._FWD: setattr(self,k,None if k=='fin' else {})
        return st
    def unstash(self,st):
        for k,v in st.items(): setattr(self,k,v)
    def snapshot(self):
        """copy of the current forward's reader inputs: {'kin':{l:x}, 'min':{l:x}, 'fin':x} (late layers only)."""
        rem=None
        for r in self.REM.values(): rem=r.clone() if rem is None else rem+r.to(rem.device)
        return {'kin':{l:x.clone() for l,x in self.kin.items()},'min':{l:x.clone() for l,x in self.min.items()},'fin':None if self.fin is None else self.fin.clone(),'rem':rem,'aout':{l:x.clone() for l,x in self.AOUT.items()},'oin':{l:x.clone() for l,x in self.oin.items()}}
    def _layer_pre(self,l):
        def pre(mod,args,kwargs):
            hs=args[0] if args else kwargs.get('hidden_states')
            if hs is None: return None
            da=self.DADD.get(l)
            if da is not None:
                v,pos,sc=da; hs=hs.clone(); hs[0,pos]=hs[0,pos]+sc*v.to(hs.device,hs.dtype)
                if args: args=(hs,)+tuple(args[1:])
                else: kwargs=dict(kwargs,hidden_states=hs)
            if self.RESPOS is not None: self.RES[l]=hs[0,self.RESPOS].detach().float().cpu().clone()
            return (args,kwargs) if da is not None else None
        return pre
    def _mlp_pre(self,l):
        def pre(mod,args):
            self.min[l]=args[0][0].detach().clone()
            xs=self.PATCH.get(l,{}).get('mlp')
            if xs is None: return None
            x=self._blend(xs,args[0].clone())
            return (x,)+tuple(args[1:])
        return pre
    def _final_pre(self,mod,args):
        # self.fin must be what the final norm actually sees, so record it AFTER any final-path patch (loupes read it; the source snapshot uses it)
        xs=self.PATCH.get('final'); dl=self.PATCH.get('final_delta'); out=None
        if xs is not None: out=(self._blend(xs,args[0].clone()),)+tuple(args[1:])
        elif dl is not None: out=(self._blend(dl,args[0].clone(),fn=lambda d,s:d+s),)+tuple(args[1:])
        self.fin=(out[0] if out is not None else args[0])[0].detach().clone()
        return out
    def _blend(self,xs,x,fn=None):
        """Write source rows into x (clone of the current [1,T,d] input) at the patchable positions: the shared prefix (see _npatch),
        further restricted to self.PATCH_POS when set (e.g. query positions only). fn(dst,src)->dst lets callers blend a slice."""
        n=self._npatch(xs,x); s=xs[:n].to(x.device,x.dtype)
        if self.PATCH_POS is None: x[:,:n]=s if fn is None else fn(x[:,:n],s[None]); return x
        p=self.PATCH_POS[self.PATCH_POS<n].to(x.device)
        if len(p): x[:,p]=s[p][None] if fn is None else fn(x[:,p],s[p][None])
        return x
    def _npatch(self,src,cur):
        """positions to patch: the shared prefix of source [T_src,d] and current [1,T_cur,d], capped by PATCH_N (set to the prompt length when the current forward has a different teacher-forced tail)."""
        n=min(src.shape[0],cur.shape[1]); return n if self.PATCH_N is None else min(n,self.PATCH_N)
    def _dir_post(self,l,which):
        def post(mod,args,out):
            if which=='attn':
                self.AOUT[l]=(out[0] if isinstance(out,tuple) else out)[0].detach().clone()
                xs=self.PATCH.get(l,{}).get('aout')   # twin/source attention-sublayer OUTPUT written at the patchable positions (what this layer's heads wrote INTO those positions)
                if xs is not None:
                    y=out[0] if isinstance(out,tuple) else out; y=self._blend(xs,y.clone())
                    self.AOUT[l]=y[0].detach().clone(); out=(y,)+tuple(out[1:]) if isinstance(out,tuple) else y
                    if self.DIRS.get(l,{}).get(which) is None: return out
            u=self.DIRS.get(l,{}).get(which)
            if u is None: return None
            y=out[0] if isinstance(out,tuple) else out
            u=u.to(y.device,y.dtype); rem=(y@u)[...,None]*u[None]; self.REM[(l,which)]=rem[0].detach().clone(); y=y-rem
            return (y,)+tuple(out[1:]) if isinstance(out,tuple) else y
        return post
    def set_knock(self,hl):
        """hl: list of (layer,head) or ('g',layer)."""
        self.KNOCK.clear(); self.GKNOCK.clear()
        for l,h in hl:
            if l=='g': self.GKNOCK.add(h)
            else: self.KNOCK.setdefault(l,[]).append(h)
    def clear(self): self.DADD={}; self.KNOCK.clear(); self.GKNOCK.clear(); self.NEUR={}; self.DIRS={}; self.KDIRS={}; self.PATCH={}; self.REM={}; self.PATCH_POS=None
    def build_means(self,run,docs_ids):
        """Mean head outputs (per layer, over natural text) for mean-ablation; also GDN layer means."""
        acc={}; n={}; gacc={}; gn={}
        for ids in docs_ids:
            run(ids)
            for l,x in self.oin.items(): acc[l]=acc.get(l,0)+x.sum(0); n[l]=n.get(l,0)+x.shape[0]
            for l,x in self.GOUT.items(): gacc[l]=gacc.get(l,0)+x.sum(0); gn[l]=gn.get(l,0)+x.shape[0]
        self.MEANO={l:acc[l]/n[l] for l in acc}; self.MEANG={l:gacc[l]/gn[l] for l in gacc}
        assert all(torch.isfinite(v).all() for v in self.MEANO.values()), 'non-finite head means'
        return self

def down_matrix(lay):
    """[d, ffn] matrix of the MLP down projection. Linear stores [out=d, in=ffn]; GPT-2 Conv1D stores [in=ffn, out=d]."""
    mlp=lay.mlp; down=getattr(mlp,'down_proj',None) or getattr(mlp,'c_proj',None) or getattr(mlp,'dense_4h_to_h',None); w=down.weight.float()
    return w.T if type(down).__name__=='Conv1D' else w
def qkv_pre(H,l):
    """Pre-norm, pre-RoPE per-head queries/keys/values at layer l in [T,heads,hd] (kproj models: recomputed from the k_proj input; gpt2/neox: the fused projection output).
    Caveat: for QK-norm/RoPE models these are NOT the exact dot-product operands; use for decomposition of what the projections read, not for reproducing attention weights."""
    lay=H.layers[l]; a=attmod(lay); st=kstyle(lay); nh,hd,nkv=H.nh,H.hd,H.nkv
    if st=='kproj':
        x=H.kin[l].float(); q=(x@a.q_proj.weight.float().T).view(-1,nh,hd); k=(x@a.k_proj.weight.float().T).view(-1,nkv,hd); v=(x@a.v_proj.weight.float().T).view(-1,nkv,hd); return q,k,v
    return H.qn[l],H.kn[l],H.vin[l]
def key_dir(H,l,u):
    """Flag direction mapped into per-KV-head key space: kproj -> Wk u per head; gpt2/neox -> u already lives in key space (split per head)."""
    lay=H.layers[l]; a=attmod(lay); nkv,hd=H.nkv,H.hd
    uk=(a.k_proj.weight.float()@u.float().to(a.k_proj.weight.device)) if kstyle(lay)=='kproj' else u.float()
    return uk.view(nkv,hd)

def norm_gain(lay):
    """weight of the pre-attention norm of a layer (post-norm direction u ↔ residual direction u/gain)."""
    n=getattr(lay,'input_layernorm',None) or getattr(lay,'ln_1',None)
    return n.weight.detach().float()
