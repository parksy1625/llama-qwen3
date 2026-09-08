from dataclasses import dataclass
import math, torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class Mini200V053Config:
    vocab_size:int=49152
    hidden_size:int=576
    base_intermediate_size:int=1536
    num_hidden_layers:int=30
    num_attention_heads:int=9
    num_key_value_heads:int=3
    head_dim:int=64
    rope_theta:float=100000.0
    rms_norm_eps:float=1e-5
    max_position_embeddings:int=8192
    insert_after_layer:int=20
    workspace_slots:int=4
    workspace_layers:int=11
    workspace_intermediate_size:int=2304
    max_recurrences:int=8
    workspace_init_scale:float=0.10
    readout_init:float=0.0
    tie_word_embeddings:bool=True

class RMSNorm(nn.Module):
    def __init__(self,d,eps=1e-5):
        super().__init__(); self.weight=nn.Parameter(torch.ones(d)); self.eps=eps
    def forward(self,x):
        y=x.float(); y=y*torch.rsqrt(y.pow(2).mean(-1,keepdim=True)+self.eps)
        return y.to(x.dtype)*self.weight

def rotate_half(x):
    h=x.shape[-1]//2
    return torch.cat((-x[...,h:],x[...,:h]),-1)

def apply_rope(q,k,theta):
    T=q.shape[-2]; D=q.shape[-1]
    p=torch.arange(T,device=q.device,dtype=torch.float32)
    inv=1.0/(theta**(torch.arange(0,D,2,device=q.device,dtype=torch.float32)/D))
    f=torch.outer(p,inv); e=torch.cat((f,f),-1)
    c=e.cos()[None,None].to(q.dtype); s=e.sin()[None,None].to(q.dtype)
    return q*c+rotate_half(q)*s,k*c+rotate_half(k)*s

class Attention(nn.Module):
    def __init__(self,cfg):
        super().__init__(); d=cfg.hidden_size; kv=cfg.num_key_value_heads*cfg.head_dim
        self.q=nn.Linear(d,d,bias=False); self.k=nn.Linear(d,kv,bias=False); self.v=nn.Linear(d,kv,bias=False); self.o=nn.Linear(d,d,bias=False)
        self.nq=cfg.num_attention_heads; self.nkv=cfg.num_key_value_heads; self.hd=cfg.head_dim; self.theta=cfg.rope_theta
    def forward(self,x):
        B,T,_=x.shape
        q=self.q(x).view(B,T,self.nq,self.hd).transpose(1,2)
        k=self.k(x).view(B,T,self.nkv,self.hd).transpose(1,2)
        v=self.v(x).view(B,T,self.nkv,self.hd).transpose(1,2)
        q,k=apply_rope(q,k,self.theta)
        rep=self.nq//self.nkv
        if rep>1:
            k=k.repeat_interleave(rep,1); v=v.repeat_interleave(rep,1)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.o(y.transpose(1,2).reshape(B,T,-1))

class MLP(nn.Module):
    def __init__(self,d,ff):
        super().__init__(); self.g=nn.Linear(d,ff,bias=False); self.u=nn.Linear(d,ff,bias=False); self.d=nn.Linear(ff,d,bias=False)
    def forward(self,x): return self.d(F.silu(self.g(x))*self.u(x))

class Block(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.n1=RMSNorm(cfg.hidden_size,cfg.rms_norm_eps); self.a=Attention(cfg); self.n2=RMSNorm(cfg.hidden_size,cfg.rms_norm_eps); self.m=MLP(cfg.hidden_size,cfg.base_intermediate_size)
    def forward(self,x):
        x=x+self.a(self.n1(x)); return x+self.m(self.n2(x))

class GQACrossAttention(nn.Module):
    def __init__(self,cfg):
        super().__init__(); d=cfg.hidden_size; kv=cfg.num_key_value_heads*cfg.head_dim
        self.q=nn.Linear(d,d,bias=False); self.k=nn.Linear(d,kv,bias=False); self.v=nn.Linear(d,kv,bias=False); self.o=nn.Linear(d,d,bias=False)
        self.qn=RMSNorm(cfg.head_dim,cfg.rms_norm_eps); self.kn=RMSNorm(cfg.head_dim,cfg.rms_norm_eps)
        self.nq=cfg.num_attention_heads; self.nkv=cfg.num_key_value_heads; self.hd=cfg.head_dim
    def forward(self,qx,kvx):
        B,S,_=qx.shape; T=kvx.shape[1]
        q=self.q(qx).view(B,S,self.nq,self.hd).transpose(1,2)
        k=self.k(kvx).view(B,T,self.nkv,self.hd).transpose(1,2)
        v=self.v(kvx).view(B,T,self.nkv,self.hd).transpose(1,2)
        q=self.qn(q); k=self.kn(k); rep=self.nq//self.nkv
        if rep>1:
            k=k.repeat_interleave(rep,1); v=v.repeat_interleave(rep,1)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=False)
        return self.o(y.transpose(1,2).reshape(B,S,-1))

class GQASelfAttention(nn.Module):
    def __init__(self,cfg):
        super().__init__(); d=cfg.hidden_size; kv=cfg.num_key_value_heads*cfg.head_dim
        self.q=nn.Linear(d,d,bias=False); self.k=nn.Linear(d,kv,bias=False); self.v=nn.Linear(d,kv,bias=False); self.o=nn.Linear(d,d,bias=False)
        self.qn=RMSNorm(cfg.head_dim,cfg.rms_norm_eps); self.kn=RMSNorm(cfg.head_dim,cfg.rms_norm_eps)
        self.nq=cfg.num_attention_heads; self.nkv=cfg.num_key_value_heads; self.hd=cfg.head_dim
    def forward(self,x):
        B,S,_=x.shape
        q=self.q(x).view(B,S,self.nq,self.hd).transpose(1,2)
        k=self.k(x).view(B,S,self.nkv,self.hd).transpose(1,2)
        v=self.v(x).view(B,S,self.nkv,self.hd).transpose(1,2)
        q=self.qn(q); k=self.kn(k); rep=self.nq//self.nkv
        if rep>1:
            k=k.repeat_interleave(rep,1); v=v.repeat_interleave(rep,1)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=False)
        return self.o(y.transpose(1,2).reshape(B,S,-1))

class WorkspaceBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__(); d=cfg.hidden_size
        self.nq=RMSNorm(d,cfg.rms_norm_eps); self.nkv=RMSNorm(d,cfg.rms_norm_eps); self.cross=GQACrossAttention(cfg)
        self.ns=RMSNorm(d,cfg.rms_norm_eps); self.selfa=GQASelfAttention(cfg)
        self.nf=RMSNorm(d,cfg.rms_norm_eps); self.ff=MLP(d,cfg.workspace_intermediate_size)
        self.g_cross=nn.Parameter(torch.tensor(-1.5)); self.g_self=nn.Parameter(torch.tensor(-1.5)); self.g_ff=nn.Parameter(torch.tensor(-1.5))
    def forward(self,w,anchor):
        w=w+torch.sigmoid(self.g_cross)*self.cross(self.nq(w),self.nkv(anchor))
        w=w+torch.sigmoid(self.g_self)*self.selfa(self.ns(w))
        w=w+torch.sigmoid(self.g_ff)*self.ff(self.nf(w))
        return w

class PersistentWorkspaceSidecar(nn.Module):
    def __init__(self,cfg):
        super().__init__(); d=cfg.hidden_size; s=cfg.workspace_slots; self.cfg=cfg
        self.anchor_norm=RMSNorm(d,cfg.rms_norm_eps)
        self.seed_proj=nn.Linear(d,d,bias=False)
        self.slots=nn.Parameter(torch.randn(1,s,d)*0.02)
        self.blocks=nn.ModuleList([WorkspaceBlock(cfg) for _ in range(cfg.workspace_layers)])
        self.read_norm=RMSNorm(d,cfg.rms_norm_eps); self.read_proj=nn.Linear(d,d,bias=False)
        self.read_gate=nn.Parameter(torch.tensor(cfg.readout_init))
    def forward_depths(self,anchor,R):
        seed=self.seed_proj(self.anchor_norm(anchor[:,-1:]))
        w=self.slots.expand(anchor.size(0),-1,-1)+self.cfg.workspace_init_scale*seed
        outs=[]
        for _ in range(R):
            for b in self.blocks: w=b(w,anchor)
            delta=self.read_proj(self.read_norm(w[:,0]))
            h=anchor.clone(); h[:,-1]=h[:,-1]+torch.tanh(self.read_gate)*delta
            outs.append(h)
        return outs

class Mini200V053(nn.Module):
    def __init__(self,cfg=Mini200V053Config()):
        super().__init__(); self.cfg=cfg; assert cfg.hidden_size==cfg.num_attention_heads*cfg.head_dim
        self.embed_tokens=nn.Embedding(cfg.vocab_size,cfg.hidden_size)
        self.layers=nn.ModuleList([Block(cfg) for _ in range(cfg.num_hidden_layers)])
        self.sidecar=PersistentWorkspaceSidecar(cfg)
        self.norm=RMSNorm(cfg.hidden_size,cfg.rms_norm_eps)
        self.lm_head=nn.Linear(cfg.hidden_size,cfg.vocab_size,bias=False)
        if cfg.tie_word_embeddings: self.lm_head.weight=self.embed_tokens.weight
    def forward_hidden_depths(self,input_ids,recurrences=1):
        R=max(1,min(int(recurrences),self.cfg.max_recurrences)); x=self.embed_tokens(input_ids)
        for i,b in enumerate(self.layers):
            x=b(x)
            if i+1==self.cfg.insert_after_layer:
                hs=self.sidecar.forward_depths(x,R)
                for j in range(i+1,len(self.layers)): hs=[self.layers[j](h) for h in hs]
                return [self.norm(h) for h in hs]
        return [self.norm(x)]
    def forward(self,input_ids,recurrences=1,return_all_depths=False):
        logits=[self.lm_head(h) for h in self.forward_hidden_depths(input_ids,recurrences)]
        return logits if return_all_depths else logits[-1]
    @torch.no_grad()
    def calibrated_mcq_scores(self,input_ids,label_token_ids,recurrences=1,null_input_ids=None):
        s=self(input_ids,recurrences)[:,-1]; ids=torch.as_tensor(label_token_ids,device=s.device); s=s.index_select(-1,ids)
        if null_input_ids is not None: s=s-self(null_input_ids,recurrences)[:,-1].index_select(-1,ids)
        return s

def count_parameters(m): return sum(p.numel() for p in m.parameters())
def architecture_report(cfg=None):
    cfg=cfg or Mini200V053Config()
    with torch.device('meta'): m=Mini200V053(cfg)
    total=count_parameters(m); side=sum(p.numel() for n,p in m.named_parameters() if n.startswith('sidecar.'))
    return {'total':total,'base_plus_head':total-side,'sidecar':side,'headroom_to_200m':200_000_000-total}

if __name__=='__main__': print(architecture_report())
