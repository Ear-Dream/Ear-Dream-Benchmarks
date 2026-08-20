from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .baseline_model import Block,zero_pad

class HybridHandModel(nn.Module):
    def __init__(self,num_classes=300,num_onehand=106,d_model=256,layers=6,heads=8,ffn=1024,kernel=15,dropout=.1,embedding_dim=128,max_len=256):
        super().__init__(); self.pose=nn.Conv1d(50,64,3,padding=1);self.hand=nn.Conv1d(42,64,3,padding=1);self.face=nn.Conv1d(74,64,3,padding=1)
        self.norms=nn.ModuleList([nn.LayerNorm(64) for _ in range(4)]);self.missing=nn.Parameter(torch.zeros(2,64));self.part_fuse=nn.Linear(256,d_model)
        self.all=nn.Conv1d(208,d_model,3,padding=1);self.all_norm=nn.LayerNorm(d_model);self.gate=nn.Sequential(nn.Linear(4,d_model),nn.Sigmoid());self.pos=nn.Parameter(torch.zeros(1,max_len,d_model))
        self.blocks=nn.ModuleList([Block(d_model,heads,ffn,kernel,dropout) for _ in range(layers)]);self.pool=nn.Sequential(nn.Linear(d_model,128),nn.Tanh(),nn.Linear(128,1));self.out=nn.LayerNorm(d_model)
        self.full_classifier=nn.Linear(d_model,num_classes);self.onehand_classifier=nn.Linear(d_model,num_onehand);self.hand_type_classifier=nn.Linear(d_model,2);self.embedding=nn.Linear(d_model,embedding_dim);nn.init.trunc_normal_(self.pos,std=.02)
    def forward(self,x,padding,detected,view):
        valid=detected.float()*view.float();inputs=(x[...,:50],x[...,50:92],x[...,92:134],x[...,134:]);convs=(self.pose,self.hand,self.hand,self.face);zs=[]
        for i,(part,conv,norm) in enumerate(zip(inputs,convs,self.norms)):
            z=F.silu(norm(conv(part.transpose(1,2)).transpose(1,2)))
            if i in (1,2):z=z*valid[...,i-1,None]+self.missing[i-1]*(1-valid[...,i-1,None])
            zs.append(z)
        parts=self.part_fuse(torch.cat(zs,-1));allz=F.silu(self.all_norm(self.all(x.transpose(1,2)).transpose(1,2)));coverage=torch.cat((valid,valid.mean(1,keepdim=True).expand_as(valid)),-1)
        z=zero_pad(parts+self.gate(coverage)*allz+self.pos[:,:x.shape[1]],padding)
        for block in self.blocks:z=block(z,padding)
        score=self.pool(z).squeeze(-1).masked_fill(padding,float("-inf"));h=self.out((z*score.softmax(1)[...,None]).sum(1))
        return {"full_logits":self.full_classifier(h),"onehand_logits":self.onehand_classifier(h),"hand_type_logits":self.hand_type_classifier(h),"embedding":F.normalize(self.embedding(h),dim=-1),"pooled":h}

def make_model(cfg,num_onehand=None):
    m=cfg["model"]
    if num_onehand is None:num_onehand=m.get("num_onehand",106)
    return HybridHandModel(m["num_classes"],num_onehand,m["d_model"],m["encoder_layers"],m["attention_heads"],m["ffn_dim"],m["conv_kernel_size"],m["dropout"],m["embedding_dim"],cfg["features"]["max_len"])
