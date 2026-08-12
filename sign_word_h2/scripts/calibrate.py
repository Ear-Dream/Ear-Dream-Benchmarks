from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.common import dump_json,load_config,resolve
from src.data import SignH5Dataset,collate_sign
from scripts.evaluate import build_model,collect

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args(); cfg=load_config(a.config)
    data=resolve(cfg,cfg["paths"]["workspace_data"]); classes=json.loads((data/"classes.json").read_text(encoding="utf-8")); run=resolve(cfg,cfg["paths"]["runs"])/(cfg["experiment"]["name"]+"_train")
    model=build_model(cfg,classes,run/"best.pt"); ds=SignH5Dataset(data/"samples.csv","val",cfg["data"]["max_sequence_length"]); loader=DataLoader(ds,batch_size=cfg["training"]["batch_size"],num_workers=cfg["data"]["num_workers"],pin_memory=True,persistent_workers=True,collate_fn=collate_sign)
    logits,labels,*_=collect(model,loader); logits=logits.cuda(); labels=labels.cuda(); log_t=torch.zeros((),device="cuda",requires_grad=True); opt=torch.optim.LBFGS([log_t],lr=0.1,max_iter=50)
    def closure():
        opt.zero_grad(); loss=torch.nn.functional.cross_entropy(logits/log_t.exp(),labels); loss.backward(); return loss
    before=float(torch.nn.functional.cross_entropy(logits,labels)); opt.step(closure); temp=float(log_t.exp().detach()); after=float(torch.nn.functional.cross_entropy(logits/temp,labels))
    confidence=(logits/temp).softmax(1).max(1).values; correct=((logits/temp).argmax(1)==labels)
    thresholds=[]
    for t in [x/100 for x in range(50,96,5)]:
        keep=confidence>=t; thresholds.append({"threshold":t,"coverage":float(keep.float().mean()),"accuracy":float(correct[keep].float().mean()) if keep.any() else None})
    result={"temperature":temp,"validation_nll_before":before,"validation_nll_after":after,"threshold_sweep":thresholds}; dump_json(run/"calibration.json",result); print(json.dumps(result))
if __name__=="__main__": main()
