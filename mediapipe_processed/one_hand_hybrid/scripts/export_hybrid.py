from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import load_config
from src.models.hybrid_model import make_model
class Wrapper(torch.nn.Module):
    def __init__(self,m):super().__init__();self.m=m
    def forward(self,x,padding,detected,view):
        o=self.m(x,padding,detected,view);return o["full_logits"],o["onehand_logits"],o["hand_type_logits"],o["embedding"]
def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--checkpoint",required=True);a=p.parse_args();cfg=load_config(a.config);m=make_model(cfg);m.load_state_dict(torch.load(a.checkpoint,map_location="cpu",weights_only=False)["model_state"]);m.eval();w=Wrapper(m);example=(torch.zeros(1,16,208),torch.zeros(1,16,dtype=torch.bool),torch.ones(1,16,2),torch.ones(1,16,2));traced=torch.jit.trace(w,example,check_trace=False);out=Path(a.checkpoint).parent/"hybrid_model_torchscript.pt";traced.save(str(out));diffs={}
    with torch.no_grad():
        for n in (16,31):
            sample=(torch.randn(2,n,208),torch.zeros(2,n,dtype=torch.bool),torch.ones(2,n,2),torch.ones(2,n,2));e=w(*sample);t=traced(*sample);diffs[str(n)]=max((x-y).abs().max().item() for x,y in zip(e,t))
    (out.parent/"hybrid_export_parity.json").write_text(json.dumps({"max_abs_diff":max(diffs.values()),"by_sequence_length":diffs},indent=2),encoding="utf-8");print(json.dumps({"model":str(out),"max_abs_diff":max(diffs.values())}))
if __name__=="__main__":main()
