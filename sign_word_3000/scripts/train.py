from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import dump_json, load_config, resolve, seed_everything
from src.data import SignH5Dataset, collate_sign
from src.model import SPOTER208


def loader(ds, cfg, shuffle):
    return DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=shuffle,
                      num_workers=cfg["data"]["num_workers"], pin_memory=cfg["data"]["pin_memory"],
                      persistent_workers=cfg["data"]["num_workers"] > 0, collate_fn=collate_sign)


@torch.no_grad()
def evaluate(model, batches, device, num_classes):
    model.eval(); total = correct = 0; loss_sum = 0.0
    class_total = torch.zeros(num_classes); class_correct = torch.zeros(num_classes)
    for batch in batches:
        x=batch["features"].to(device, non_blocking=True); mask=batch["padding_mask"].to(device, non_blocking=True)
        y=batch["labels"].to(device, non_blocking=True); logits=model(x, mask)
        loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        pred=logits.argmax(1); total += len(y); correct += (pred==y).sum().item()
        class_total.scatter_add_(0, y.cpu(), torch.ones_like(y.cpu(), dtype=torch.float32))
        class_correct.scatter_add_(0, y.cpu(), (pred==y).cpu().float())
    macro=((class_correct/class_total.clamp_min(1))[class_total>0].mean().item())
    return {"loss":loss_sum/total, "micro_top1":correct/total, "macro_top1":macro}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--mode",choices=["overfit32","smoke10","train"],default="train"); args=parser.parse_args()
    cfg=load_config(args.config); seed=cfg["experiment"]["seed"]; seed_everything(seed)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required; refusing accidental CPU training")
    device=torch.device("cuda:0"); data_dir=resolve(cfg,cfg["paths"]["workspace_data"])
    manifest=data_dir/"samples.csv"; classes=json.loads((data_dir/"classes.json").read_text(encoding="utf-8")); full_classes=len(classes)
    labels=None; epochs=cfg["training"]["epochs"]; dropout=cfg["model"]["dropout"]
    train_ds=SignH5Dataset(manifest,"train",cfg["data"]["max_sequence_length"])
    val_ds=SignH5Dataset(manifest,"val",cfg["data"]["max_sequence_length"])
    if args.mode=="overfit32":
        train_ds=Subset(train_ds,list(range(32))); val_ds=train_ds; epochs=200; dropout=0.0; num_classes=full_classes
    elif args.mode=="smoke10":
        labels=set(range(10)); train_ds=SignH5Dataset(manifest,"train",cfg["data"]["max_sequence_length"],labels); val_ds=SignH5Dataset(manifest,"val",cfg["data"]["max_sequence_length"],labels); epochs=5; num_classes=10
    else: num_classes=full_classes
    m=cfg["model"]; model=SPOTER208(input_dim=m["input_dim"],d_model=m["d_model"],nhead=m["nhead"],encoder_layers=m["encoder_layers"],decoder_layers=m["decoder_layers"],dim_feedforward=m["dim_feedforward"],dropout=dropout,activation=m["activation"],max_sequence_length=cfg["data"]["max_sequence_length"],num_classes=num_classes).to(device)
    transfer_path=cfg.get("paths",{}).get("transfer_checkpoint")
    if args.mode=="train" and transfer_path:
        candidate=resolve(cfg,transfer_path)
        run_candidate=resolve(cfg,cfg["paths"]["runs"])/(cfg["experiment"]["name"]+"_"+args.mode)/"last.pt"
        if candidate.exists() and not run_candidate.exists():
            source=torch.load(candidate,map_location=device,weights_only=False)["model_state"]
            target=model.state_dict(); reusable={k:v for k,v in source.items() if k in target and target[k].shape==v.shape and not k.startswith("classifier.")}
            missing,unexpected=model.load_state_dict(reusable,strict=False)
            print(json.dumps({"transfer_checkpoint":str(candidate),"loaded_tensors":len(reusable),"new_classifier":num_classes,"missing":missing,"unexpected":unexpected}),flush=True)
    optim=torch.optim.AdamW(model.parameters(),lr=cfg["training"]["learning_rate"],weight_decay=cfg["training"]["weight_decay"])
    warmup=max(1,cfg["training"]["warmup_epochs"])
    def lr_factor(epoch):
        if epoch < warmup: return float(epoch+1)/warmup
        progress=(epoch-warmup)/max(1,epochs-warmup)
        return 0.5*(1.0+math.cos(math.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optim,lr_factor)
    scaler=torch.amp.GradScaler("cuda",enabled=True); train_loader=loader(train_ds,cfg,True); val_loader=loader(val_ds,cfg,False)
    run_dir=resolve(cfg,cfg["paths"]["runs"])/(cfg["experiment"]["name"]+"_"+args.mode); run_dir.mkdir(parents=True,exist_ok=True)
    shutil.copy2(args.config,run_dir/"config.yaml"); shutil.copy2(data_dir/"splits.json",run_dir/"splits.json"); shutil.copy2(data_dir/"classes.json",run_dir/"classes.json"); shutil.copy2(data_dir/"manifest_hash.txt",run_dir/"manifest_hash.txt")
    start=0; best=-1.0; no_improve=0; last=run_dir/"last.pt"; history_path=run_dir/"history.csv"
    if cfg["training"].get("resume") and last.exists():
        ck=torch.load(last,map_location=device,weights_only=False); model.load_state_dict(ck["model_state"]); optim.load_state_dict(ck["optimizer_state"]); scaler.load_state_dict(ck["grad_scaler_state"]); start=ck["epoch"]+1; best=ck["best_metric"]
        if "scheduler_state" in ck: scheduler.load_state_dict(ck["scheduler_state"])
        no_improve=int(ck.get("no_improve",0))
        if "no_improve" not in ck and history_path.exists():
            with history_path.open("r",encoding="utf-8",newline="") as old:
                rows=list(csv.DictReader(old))
            if rows:
                scores=[float(r["val_macro_top1"]) for r in rows]
                last_best=scores.index(max(scores))
                no_improve=len(scores)-1-last_best
    write_header=not history_path.exists()
    if args.mode=="train" and no_improve>=cfg["training"]["early_stopping_patience"]:
        start=epochs
    with history_path.open("a",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["epoch","train_loss","train_acc","val_loss","val_micro_top1","val_macro_top1","seconds"])
        if write_header: writer.writeheader()
        for epoch in range(start,epochs):
            model.train(); seen=correct=0; loss_sum=0.0; tick=time.time()
            for batch in train_loader:
                x=batch["features"].to(device,non_blocking=True); mask=batch["padding_mask"].to(device,non_blocking=True); y=batch["labels"].to(device,non_blocking=True)
                optim.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.float16): logits=model(x,mask); loss=nn.functional.cross_entropy(logits,y)
                scaler.scale(loss).backward(); scaler.unscale_(optim); nn.utils.clip_grad_norm_(model.parameters(),cfg["training"]["gradient_clip_norm"]); scaler.step(optim); scaler.update()
                seen+=len(y); loss_sum+=loss.item()*len(y); correct+=(logits.argmax(1)==y).sum().item()
            val=evaluate(model,val_loader,device,num_classes); record={"epoch":epoch,"train_loss":loss_sum/seen,"train_acc":correct/seen,"val_loss":val["loss"],"val_micro_top1":val["micro_top1"],"val_macro_top1":val["macro_top1"],"seconds":time.time()-tick}; writer.writerow(record); stream.flush(); print(json.dumps(record),flush=True)
            scheduler.step()
            improved=val["macro_top1"]>best
            if improved: best=val["macro_top1"]; no_improve=0
            else: no_improve+=1
            state={"model_state":model.state_dict(),"optimizer_state":optim.state_dict(),"scheduler_state":scheduler.state_dict(),"grad_scaler_state":scaler.state_dict(),"epoch":epoch,"best_metric":best,"no_improve":no_improve,"config":cfg,"feature_version":cfg["data"]["feature_version"],"label_mapping":classes}
            torch.save(state,last)
            if improved: torch.save(state,run_dir/"best.pt")
            if args.mode=="overfit32" and record["train_acc"]>=0.95: break
            if args.mode=="train" and no_improve>=cfg["training"]["early_stopping_patience"]: break
    dump_json(run_dir/"metrics.json",{"best_val_macro_top1":best,"mode":args.mode})


if __name__=="__main__": main()
