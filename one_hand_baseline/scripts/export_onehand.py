from __future__ import annotations
import argparse,json,shutil,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.common import load_config
from src.models.a1p_mask_aware import make_model
class Wrapper(torch.nn.Module):
    def __init__(self,m): super().__init__(); self.m=m
    def forward(self,x,padding,detected,view):
        o=self.m(x,padding,detected,view); return o["logits"],o["embedding"]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--prototypes"); a=p.parse_args(); cfg=load_config(a.config); m=make_model(cfg); m.load_state_dict(torch.load(a.checkpoint,map_location="cpu",weights_only=False)["model_state"]); m.eval(); w=Wrapper(m)
    t=16; example=(torch.zeros(1,t,208),torch.zeros(1,t,dtype=torch.bool),torch.ones(1,t,2),torch.ones(1,t,2))
    # PyTorch 2.11 MHA may select a different optimized graph on the trace check
    # invocation. Freeze one inference graph, then verify values explicitly.
    traced=torch.jit.trace(w,example,check_trace=False); out=Path(a.checkpoint).parent/"onehand_model_torchscript.pt"; traced.save(str(out))
    diffs={}
    with torch.no_grad():
        for length in (16,31):
            sample=(torch.randn(2,length,208),torch.zeros(2,length,dtype=torch.bool),torch.ones(2,length,2),torch.ones(2,length,2))
            eager=w(*sample); scripted=traced(*sample); diffs[str(length)]=max((x-y).abs().max().item() for x,y in zip(eager,scripted))
    diff=max(diffs.values()); (out.parent/"export_parity.json").write_text(json.dumps({"max_abs_diff":diff,"by_sequence_length":diffs},indent=2),encoding="utf-8")
    if a.prototypes:
        source=Path(a.prototypes).resolve(); target=(out.parent/"prototypes.npz").resolve()
        if source!=target: shutil.copy2(source,target)
    print(json.dumps({"model":str(out),"max_abs_diff":diff}))
if __name__=="__main__":main()
