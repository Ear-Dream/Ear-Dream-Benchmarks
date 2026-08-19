from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

SLICES=((0,50),(50,92),(92,134),(134,208)); DIMS=(50,42,42,74)
def zero_pad(x,p): return x.masked_fill(p[...,None],0) if p is not None else x

class Block(nn.Module):
    def __init__(self,d=256,heads=8,ff=1024,kernel=15,drop=.1):
        super().__init__(); self.n1=nn.LayerNorm(d); self.a=nn.MultiheadAttention(d,heads,dropout=drop,batch_first=True)
        self.n2=nn.LayerNorm(d); self.ff=nn.Sequential(nn.Linear(d,ff),nn.SiLU(),nn.Dropout(drop),nn.Linear(ff,d))
        self.n3=nn.LayerNorm(d); self.p1=nn.Conv1d(d,d*2,1); self.dw=nn.Conv1d(d,d,kernel,padding=kernel//2,groups=d); self.p2=nn.Conv1d(d,d,1); self.out=nn.LayerNorm(d)
    def forward(self,x,p):
        q=self.n1(x); z,_=self.a(q,q,q,key_padding_mask=p,need_weights=False); x=zero_pad(x+z,p); x=zero_pad(x+self.ff(self.n2(x)),p)
        z=F.glu(self.p1(self.n3(x).transpose(1,2)),dim=1); z=self.p2(F.silu(self.dw(z))).transpose(1,2)
        return zero_pad(self.out(x+z),p)

class MaskAwareCandidateModel(nn.Module):
    def __init__(self,num_classes=300,d_model=256,layers=6,heads=8,ffn_dim=1024,kernel=15,dropout=.1,embedding_dim=128,max_len=256):
        super().__init__(); self.part=nn.ModuleList([nn.Conv1d(d,64,3,padding=1) for d in DIMS]); self.part_norm=nn.ModuleList([nn.LayerNorm(64) for _ in DIMS])
        self.part_fuse=nn.Linear(256,d_model); self.all=nn.Conv1d(208,d_model,3,padding=1); self.all_norm=nn.LayerNorm(d_model)
        self.missing=nn.Parameter(torch.zeros(2,64)); self.gate=nn.Sequential(nn.Linear(4,d_model),nn.Sigmoid()); self.pos=nn.Parameter(torch.zeros(1,max_len,d_model))
        self.blocks=nn.ModuleList([Block(d_model,heads,ffn_dim,kernel,dropout) for _ in range(layers)])
        self.pool=nn.Sequential(nn.Linear(d_model,128),nn.Tanh(),nn.Linear(128,1)); self.norm=nn.LayerNorm(d_model)
        self.classifier=nn.Linear(d_model,num_classes); self.embedding=nn.Linear(d_model,embedding_dim)
        nn.init.trunc_normal_(self.pos,std=.02)
    def forward(self,x,padding_mask,detected_mask,view_mask):
        valid=detected_mask.float()*view_mask.float(); zs=[]
        for i,((a,b),conv,norm) in enumerate(zip(SLICES,self.part,self.part_norm)):
            z=F.silu(norm(conv(x[...,a:b].transpose(1,2)).transpose(1,2)))
            if i in (1,2): z=z*valid[...,i-1,None]+self.missing[i-1]* (1-valid[...,i-1,None])
            zs.append(z)
        parts=self.part_fuse(torch.cat(zs,-1)); all_z=F.silu(self.all_norm(self.all(x.transpose(1,2)).transpose(1,2)))
        coverage=torch.cat((valid,valid.mean(1,keepdim=True).expand_as(valid)),dim=-1)
        z=parts+self.gate(coverage)*all_z; z=zero_pad(z+self.pos[:,:x.shape[1]],padding_mask)
        for block in self.blocks: z=block(z,padding_mask)
        score=self.pool(z).squeeze(-1).masked_fill(padding_mask,float("-inf")); pooled=self.norm((z*score.softmax(1)[...,None]).sum(1))
        return {"logits":self.classifier(pooled),"embedding":F.normalize(self.embedding(pooled),dim=-1),"pooled":pooled}

def make_model(cfg,num_classes=None):
    m=cfg["model"]
    return MaskAwareCandidateModel(num_classes or m["num_classes"],m["d_model"],m["encoder_layers"],m["attention_heads"],m["ffn_dim"],m["conv_kernel_size"],m["dropout"],m["candidate_embedding_dim"],cfg["features"]["max_len"])
