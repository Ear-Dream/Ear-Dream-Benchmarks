import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.model import HybridClassifier,PartFrontend

def main():
    x=torch.randn(2,17,208)
    rebuilt=torch.cat([x[...,a:b] for a,b in PartFrontend.SLICES],-1)
    assert torch.equal(x,rebuilt)
    for variant in ("h1","h2","h3"):
        model=HybridClassifier(variant=variant,d_model=256,nhead=8,encoder_layers=1,decoder_layers=1,dim_feedforward=256,conv_kernel_size=15,dropout=0.0,max_sequence_length=64,num_classes=10).eval()
        mask=torch.zeros(2,17,dtype=torch.bool)
        padded=torch.cat([x,torch.zeros(2,11,208)],1); pm=torch.cat([mask,torch.ones(2,11,dtype=torch.bool)],1)
        with torch.no_grad(): a=model(x,mask); b=model(padded,pm)
        assert a.shape==(2,10)
        assert torch.allclose(a,b,atol=2e-5,rtol=2e-4),(variant,(a-b).abs().max())
    model=HybridClassifier(variant="h3",d_model=256,nhead=8,encoder_layers=1,decoder_layers=1,dim_feedforward=256,conv_kernel_size=15,dropout=0.0,max_sequence_length=64,num_classes=10,head="attention_pooling").eval()
    mask=torch.zeros(2,17,dtype=torch.bool); padded=torch.cat([x,torch.zeros(2,11,208)],1); pm=torch.cat([mask,torch.ones(2,11,dtype=torch.bool)],1)
    with torch.no_grad():
        a=model(x,mask); b=model(padded,pm); memory=model.encode(padded,pm)
        score=model.pool_score(memory).squeeze(-1).masked_fill(pm,float("-inf")); weight=score.softmax(1)
    assert torch.allclose(a,b,atol=2e-5,rtol=2e-4),(a-b).abs().max()
    assert torch.equal(weight[pm],torch.zeros_like(weight[pm]))
    assert torch.allclose(weight.masked_fill(pm,0).sum(1),torch.ones(2),atol=1e-6)
    print("feature slicing, shapes, padding invariance, decoder/pooling masks: passed")
if __name__=="__main__": main()
