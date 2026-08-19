import torch
import torch.nn.functional as F

def supcon(z,y,temp=.1):
    sim=z@z.T/temp;eye=torch.eye(len(z),device=z.device,dtype=torch.bool);sim=sim.masked_fill(eye,-1e9);pos=(y[:,None]==y[None,:])&~eye;logp=sim-torch.logsumexp(sim,1,keepdim=True)
    return -(logp*pos).sum(1).div(pos.sum(1).clamp_min(1)).mean()
def hybrid_loss(full,partial,labels,hand_types,one_labels,w,temp=.1):
    full_ce=F.cross_entropy(full["full_logits"],labels); mask=hand_types==0
    one_ce=F.cross_entropy(partial["onehand_logits"][mask],one_labels[mask]) if mask.any() else full_ce*0
    align=(1-(partial["embedding"]*full["embedding"].detach()).sum(-1)).mean();contrast=supcon(full["embedding"],labels,temp);type_ce=F.cross_entropy(full["hand_type_logits"],hand_types)
    total=w["full_ce"]*full_ce+w["onehand_ce"]*one_ce+w["alignment"]*align+w["supcon"]*contrast+w["hand_type_ce"]*type_ce
    return total,{"full_ce":full_ce.detach(),"onehand_ce":one_ce.detach(),"alignment":align.detach(),"supcon":contrast.detach(),"hand_type_ce":type_ce.detach()}
