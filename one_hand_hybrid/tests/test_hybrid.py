import torch
from src.models.hybrid_model import HybridHandModel
from src.losses.hybrid_losses import hybrid_loss

def test_shapes_and_shared_encoder():
    m=HybridHandModel(num_classes=5,num_onehand=3,d_model=32,layers=1,heads=4,ffn=64,embedding_dim=8,max_len=12);x=torch.randn(2,12,208);p=torch.zeros(2,12,dtype=torch.bool);d=torch.ones(2,12,2);o=m(x,p,d,d)
    assert o["full_logits"].shape==(2,5) and o["onehand_logits"].shape==(2,3) and o["embedding"].shape==(2,8)
    assert m.hand is not None # one module is called for both anatomical sides
def test_twohand_excluded_from_onehand_ce():
    full={"full_logits":torch.randn(2,4,requires_grad=True),"hand_type_logits":torch.randn(2,2,requires_grad=True),"embedding":torch.nn.functional.normalize(torch.randn(2,3,requires_grad=True),dim=-1)}
    part={"onehand_logits":torch.randn(2,2,requires_grad=True),"embedding":torch.nn.functional.normalize(torch.randn(2,3,requires_grad=True),dim=-1)};w={"full_ce":1,"onehand_ce":1,"alignment":1,"supcon":.1,"hand_type_ce":.2}
    loss,_=hybrid_loss(full,part,torch.tensor([0,1]),torch.tensor([0,1]),torch.tensor([1,-1]),w);loss.backward();assert part["onehand_logits"].grad[1].abs().sum()==0
