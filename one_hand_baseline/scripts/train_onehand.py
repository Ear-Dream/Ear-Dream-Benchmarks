from __future__ import annotations
import argparse,csv,json,math,sys,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Subset
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.common import load_config,resolve,seed_everything,dump_json
from src.data.onehand_dataset import OneHandDataset,collate_onehand
from src.models.a1p_mask_aware import make_model
from src.losses.candidate_losses import candidate_loss

@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); seen=correct=0; align=0
    for b in loader:
        p=b["padding_mask"].to(device); d=b["detected_mask"].to(device); full=model(b["x_full"].to(device),p,d,b["full_view_mask"].to(device)); part=model(b["x_partial"].to(device),p,d,b["partial_view_mask"].to(device)); y=b["labels"].to(device)
        seen+=len(y); correct+=(full["logits"].argmax(1)==y).sum().item(); align+=(full["embedding"]*part["embedding"]).sum().item()
    return {"full_top1":correct/max(1,seen),"pair_cosine":align/max(1,seen)}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--mode",choices=["overfit32","smoke10","train","resume"],default="train"); a=p.parse_args(); cfg=load_config(a.config); seed_everything(cfg["experiment"]["seed"])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); manifest=resolve(cfg,cfg["paths"]["workspace_data"])/"samples.csv"; maxlen=cfg["features"]["max_len"]
    tr=OneHandDataset(manifest,"train",maxlen); va=OneHandDataset(manifest,"val",maxlen,deterministic_side="right_only"); epochs=cfg["training"]["max_epochs"]
    if a.mode=="overfit32": tr=Subset(tr,range(min(32,len(tr)))); va=tr; epochs=200
    elif a.mode=="smoke10": tr=Subset(tr,range(min(320,len(tr)))); va=Subset(va,range(min(100,len(va)))); epochs=5
    loader=lambda ds,shuffle:DataLoader(ds,batch_size=cfg["training"]["batch_size"],shuffle=shuffle,num_workers=cfg["training"]["num_workers"],collate_fn=collate_onehand,persistent_workers=cfg["training"]["num_workers"]>0)
    tl,vl=loader(tr,True),loader(va,False); model=make_model(cfg).to(device); opt=torch.optim.AdamW(model.parameters(),lr=cfg["training"]["learning_rate"],weight_decay=cfg["training"]["weight_decay"])
    run_mode="train" if a.mode=="resume" else a.mode
    run=resolve(cfg,cfg["paths"]["runs"])/(cfg["experiment"]["name"]+"_"+run_mode); run.mkdir(parents=True,exist_ok=True); start=0; best=-1.; patience=0
    last=run/"last.pt"
    if a.mode=="resume" and last.exists(): ck=torch.load(last,map_location=device,weights_only=False); model.load_state_dict(ck["model_state"]); opt.load_state_dict(ck["optimizer_state"]); start=ck["epoch"]+1; best=ck["best_metric"]
    hist=run/"history.csv"; fields=["epoch","loss","classification","alignment","supcon","full_top1","pair_cosine","seconds"]
    with hist.open("a",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        if f.tell()==0:w.writeheader()
        for epoch in range(start,epochs):
            tick=time.time(); model.train(); sums={k:0. for k in ("loss","classification","alignment","supcon")}; seen=0
            for b in tl:
                p=b["padding_mask"].to(device); d=b["detected_mask"].to(device); y=b["labels"].to(device); opt.zero_grad(set_to_none=True)
                full=model(b["x_full"].to(device),p,d,b["full_view_mask"].to(device)); part=model(b["x_partial"].to(device),p,d,b["partial_view_mask"].to(device)); loss,parts=candidate_loss(full,part,y,cfg["loss"],cfg["loss"]["temperature"])
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["training"]["gradient_clip"]); opt.step(); n=len(y); seen+=n; sums["loss"]+=loss.item()*n
                for k,v in parts.items(): sums[k]+=v.item()*n
            val=evaluate(model,vl,device); rec={"epoch":epoch,**{k:v/seen for k,v in sums.items()},**val,"seconds":time.time()-tick}; w.writerow(rec); f.flush(); print(json.dumps(rec),flush=True)
            improved=val["pair_cosine"]>best; best=max(best,val["pair_cosine"]); patience=0 if improved else patience+1
            state={"model_state":model.state_dict(),"optimizer_state":opt.state_dict(),"epoch":epoch,"best_metric":best,"config":cfg}; torch.save(state,last)
            if improved: torch.save(state,run/"best.pt")
            if a.mode=="overfit32" and val["full_top1"]>=.95: break
            if a.mode in ("train","resume") and patience>=cfg["training"]["early_stopping_patience"]: break
    dump_json(run/"metrics.json",{"best_pair_cosine":best,"device":str(device)})
if __name__=="__main__":main()
