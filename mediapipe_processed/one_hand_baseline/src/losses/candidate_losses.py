import torch
import torch.nn.functional as F

def supervised_contrastive(z,y,temp=.1):
    sim=z@z.T/temp; eye=torch.eye(len(z),device=z.device,dtype=torch.bool); sim=sim.masked_fill(eye,-1e9)
    pos=(y[:,None]==y[None,:])&~eye; logp=sim-torch.logsumexp(sim,1,keepdim=True)
    return -(logp*pos).sum(1).div(pos.sum(1).clamp_min(1)).mean()

def candidate_loss(full,partial,labels,weights,temp=.1):
    cls=F.cross_entropy(full["logits"],labels); align=(1-(partial["embedding"]*full["embedding"].detach()).sum(-1)).mean()
    sup=supervised_contrastive(full["embedding"],labels,temp)
    total=weights.get("classification",1)*cls+weights.get("partial_alignment",1)*align+weights.get("full_supcon",.1)*sup
    return total,{"classification":cls.detach(),"alignment":align.detach(),"supcon":sup.detach()}
