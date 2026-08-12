from __future__ import annotations

import argparse, io, json, sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.common import dump_json,load_config,resolve
from scripts.evaluate import build_model

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args(); cfg=load_config(a.config)
    data=resolve(cfg,cfg["paths"]["workspace_data"]); classes=json.loads((data/"classes.json").read_text(encoding="utf-8")); run=resolve(cfg,cfg["paths"]["runs"])/(cfg["experiment"]["name"]+"_train")
    model=build_model(cfg,classes,run/"best.pt").eval(); x=torch.randn(1,128,208,device="cuda"); mask=torch.zeros(1,128,dtype=torch.bool,device="cuda")
    with torch.no_grad(): traced=torch.jit.trace(model,(x,mask),strict=False)
    buffer=io.BytesIO(); torch.jit.save(traced,buffer); (run/"model_torchscript.pt").write_bytes(buffer.getvalue()); checks=[]
    for length in (64,128,187,250):
        sample=torch.randn(1,length,208,device="cuda"); padding=torch.zeros(1,length,dtype=torch.bool,device="cuda")
        with torch.no_grad(): native=model(sample,padding); exported=traced(sample,padding)
        checks.append({"length":length,"max_abs_logit_difference":float((native-exported).abs().max()),"top1_match":bool(native.argmax(1).eq(exported.argmax(1)).all()),"top5_match":bool(torch.equal(native.topk(5,1).indices,exported.topk(5,1).indices))})
    result={"format":"TorchScript","artifact":"model_torchscript.pt","checks":checks}; dump_json(run/"export_parity.json",result); print(json.dumps(result))
if __name__=="__main__": main()
