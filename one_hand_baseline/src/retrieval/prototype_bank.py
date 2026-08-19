import torch
import torch.nn.functional as F

def build_prototypes(embeddings,labels,num_classes):
    out=torch.zeros(num_classes,embeddings.shape[1],device=embeddings.device); count=torch.zeros(num_classes,device=embeddings.device)
    out.index_add_(0,labels,embeddings); count.index_add_(0,labels,torch.ones_like(labels,dtype=torch.float))
    return F.normalize(out/count.clamp_min(1)[:,None],dim=-1),count
def rank(query,prototypes): return query@prototypes.T
