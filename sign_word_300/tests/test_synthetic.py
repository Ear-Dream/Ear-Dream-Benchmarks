import torch
from sign_word_300.src.data import collate_sign
from sign_word_300.src.model import SPOTER208

def test_shapes_and_padding_invariance():
    model=SPOTER208(d_model=64,nhead=8,encoder_layers=1,decoder_layers=1,dim_feedforward=128,dropout=0.0,max_sequence_length=32,num_classes=10).eval()
    x=torch.randn(1,12,208); m=torch.zeros(1,12,dtype=torch.bool)
    padded=torch.cat([x,torch.zeros(1,5,208)],1); pm=torch.cat([m,torch.ones(1,5,dtype=torch.bool)],1)
    with torch.no_grad():
        a=model(x,m); b=model(padded,pm)
    assert a.shape==(1,10)
    assert torch.allclose(a,b,atol=1e-5,rtol=1e-4)
