import torch
from src.data.view_generator import make_view
from src.models.a1p_mask_aware import MaskAwareCandidateModel

def test_views_keep_detected_separate():
    x=torch.randn(5,208); d=torch.tensor([[1,1]]*5,dtype=torch.uint8)
    r,v,valid=make_view(x,d,"right_only")
    assert torch.count_nonzero(r[:,92:134])==0 and torch.equal(d,torch.ones_like(d))
    assert torch.equal(v[:,0],torch.ones(5,dtype=torch.uint8)) and not valid[:,1].any()

def test_forward_all_modes():
    model=MaskAwareCandidateModel(num_classes=3,d_model=32,layers=1,heads=4,ffn_dim=64,embedding_dim=8,max_len=12)
    x=torch.randn(2,12,208); p=torch.zeros(2,12,dtype=torch.bool); d=torch.ones(2,12,2)
    for mode in ("full","right_only","left_only"):
        z,v,_=make_view(x,d,mode); out=model(z,p,d,v)
        assert out["logits"].shape==(2,3) and out["embedding"].shape==(2,8)
        assert torch.isfinite(out["embedding"]).all()

def test_distractor_invariance():
    x=torch.randn(2,8,208); d=torch.ones(2,8,2); a,v,_=make_view(x,d,"right_only")
    x[...,92:134]=torch.randn_like(x[...,92:134])*100; b,_,_=make_view(x,d,"right_only")
    assert torch.equal(a,b)
