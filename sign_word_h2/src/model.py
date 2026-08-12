from __future__ import annotations

import torch
from torch import nn


def _zero_padding(x, padding_mask):
    return x.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class PartFrontend(nn.Module):
    SLICES=((0,50),(50,92),(92,134),(134,208))
    DIMS=(50,42,42,74)
    def __init__(self, hidden=64, dropout=0.1):
        super().__init__()
        self.branches=nn.ModuleList([nn.Conv1d(d,hidden,3,padding=1) for d in self.DIMS])
        self.norms=nn.ModuleList([nn.LayerNorm(hidden) for _ in self.DIMS])
        self.dropout=nn.Dropout(dropout); self.fusion=nn.Sequential(nn.Linear(hidden*4,256),nn.LayerNorm(256))
    def forward(self,x,padding_mask):
        outputs=[]
        for (start,end),conv,norm in zip(self.SLICES,self.branches,self.norms):
            z=conv(x[...,start:end].transpose(1,2)).transpose(1,2)
            z=self.dropout(torch.nn.functional.silu(norm(z))); outputs.append(_zero_padding(z,padding_mask))
        return _zero_padding(self.fusion(torch.cat(outputs,-1)),padding_mask)


class AllFrontend(nn.Module):
    def __init__(self,dropout=0.1):
        super().__init__(); self.conv=nn.Conv1d(208,256,3,padding=1); self.norm=nn.LayerNorm(256); self.dropout=nn.Dropout(dropout)
    def forward(self,x,padding_mask):
        z=self.conv(x.transpose(1,2)).transpose(1,2); z=self.dropout(torch.nn.functional.silu(self.norm(z)))
        return _zero_padding(z,padding_mask)


class SqueezeformerBlock(nn.Module):
    def __init__(self,d_model=256,nhead=8,ffn_dim=1024,kernel=15,dropout=0.1):
        super().__init__(); self.attn_norm=nn.LayerNorm(d_model); self.attn=nn.MultiheadAttention(d_model,nhead,dropout=dropout,batch_first=True); self.drop=nn.Dropout(dropout)
        self.ffn_norm=nn.LayerNorm(d_model); self.ffn=nn.Sequential(nn.Linear(d_model,ffn_dim),nn.SiLU(),nn.Dropout(dropout),nn.Linear(ffn_dim,d_model),nn.Dropout(dropout))
        self.conv_norm=nn.LayerNorm(d_model); self.point_in=nn.Conv1d(d_model,d_model*2,1); self.depth=nn.Conv1d(d_model,d_model,kernel,padding=kernel//2,groups=d_model); self.depth_norm=nn.LayerNorm(d_model); self.point_out=nn.Conv1d(d_model,d_model,1); self.out_norm=nn.LayerNorm(d_model)
    def forward(self,x,padding_mask):
        q=self.attn_norm(x); a,_=self.attn(q,q,q,key_padding_mask=padding_mask,need_weights=False); x=_zero_padding(x+self.drop(a),padding_mask)
        x=_zero_padding(x+self.ffn(self.ffn_norm(x)),padding_mask)
        z=self.conv_norm(x).transpose(1,2); z=torch.nn.functional.glu(self.point_in(z),dim=1)
        if padding_mask is not None: z=z.masked_fill(padding_mask.unsqueeze(1),0.0)
        z=self.depth(z).transpose(1,2); z=torch.nn.functional.silu(self.depth_norm(z)); z=self.point_out(z.transpose(1,2)).transpose(1,2)
        return _zero_padding(self.out_norm(x+self.drop(z)),padding_mask)


class HybridClassifier(nn.Module):
    def __init__(self,*,variant="h1",d_model=256,nhead=8,encoder_layers=6,decoder_layers=6,dim_feedforward=1024,conv_kernel_size=15,dropout=0.1,max_sequence_length=256,num_classes=300,head="decoder"):
        super().__init__(); self.variant=variant; self.head=head; self.parts=PartFrontend(64,dropout); self.all_branch=AllFrontend(dropout) if variant=="h3" else None; self.fusion_norm=nn.LayerNorm(d_model); self.position=nn.Parameter(torch.empty(1,max_sequence_length,d_model)); nn.init.trunc_normal_(self.position,std=0.02)
        if variant=="h1":
            layer=nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward,dropout,activation="gelu",batch_first=True,norm_first=True); self.encoder=nn.TransformerEncoder(layer,encoder_layers,norm=nn.LayerNorm(d_model))
        else: self.encoder=nn.ModuleList([SqueezeformerBlock(d_model,nhead,dim_feedforward,conv_kernel_size,dropout) for _ in range(encoder_layers)])
        if head=="decoder":
            dec=nn.TransformerDecoderLayer(d_model,nhead,dim_feedforward,dropout,activation="gelu",batch_first=True,norm_first=True); self.decoder=nn.TransformerDecoder(dec,decoder_layers,norm=nn.LayerNorm(d_model)); self.class_query=nn.Parameter(torch.empty(1,1,d_model)); nn.init.trunc_normal_(self.class_query,std=0.02)
        else:
            self.pool_score=nn.Sequential(nn.Linear(d_model,128),nn.Tanh(),nn.Dropout(dropout),nn.Linear(128,1))
        self.output_norm=nn.LayerNorm(d_model); self.classifier=nn.Linear(d_model,num_classes)
    def encode(self,features,padding_mask):
        z=self.parts(features,padding_mask)
        if self.all_branch is not None: z=self.fusion_norm(z+self.all_branch(features,padding_mask))
        z=_zero_padding(z+self.position[:,:z.shape[1]],padding_mask)
        if self.variant=="h1": return self.encoder(z,src_key_padding_mask=padding_mask)
        for block in self.encoder: z=block(z,padding_mask)
        return z
    def forward(self,features,padding_mask):
        memory=self.encode(features,padding_mask)
        if self.head=="decoder":
            query=self.class_query.expand(features.shape[0],-1,-1); pooled=self.decoder(query,memory,memory_key_padding_mask=padding_mask)[:,0]
        else:
            score=self.pool_score(memory).squeeze(-1).masked_fill(padding_mask,float("-inf")); weight=score.softmax(1); pooled=(memory*weight.unsqueeze(-1)).sum(1)
        return self.classifier(self.output_norm(pooled))


def make_model(cfg,num_classes,dropout_override=None):
    m=cfg["model"]; dropout=m["dropout"] if dropout_override is None else dropout_override
    return HybridClassifier(variant=m["variant"],d_model=m["d_model"],nhead=m["nhead"],encoder_layers=m["encoder_layers"],decoder_layers=m["decoder_layers"],dim_feedforward=m["dim_feedforward"],conv_kernel_size=m.get("conv_kernel_size",15),dropout=dropout,max_sequence_length=cfg["data"]["max_sequence_length"],num_classes=num_classes,head=m.get("head","decoder"))
