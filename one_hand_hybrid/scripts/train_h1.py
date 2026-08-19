from __future__ import annotations
import argparse,csv,json,sys,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Subset
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import load_config,resolve,seed_everything,dump_json
from src.data.hybrid_dataset import HybridDataset,collate_hybrid
from src.models.hybrid_model import make_model
from src.losses.hybrid_losses import hybrid_loss

def initialize_from_baseline(model,checkpoint_path,partition_path):
    old=torch.load(checkpoint_path,map_location="cpu",weights_only=False)["model_state"];new=model.state_dict();loaded=[]
    for key in list(new):
        source={"full_classifier.weight":"classifier.weight","full_classifier.bias":"classifier.bias"}.get(key,key)
        if source in old and old[source].shape==new[key].shape:new[key]=old[source];loaded.append(key)
    for name,index in (("pose",0),("face",3)):
        for suffix in ("weight","bias"):new[f"{name}.{suffix}"]=old[f"part.{index}.{suffix}"];loaded.append(f"{name}.{suffix}")
    for suffix in ("weight","bias"):
        new[f"hand.{suffix}"]=old[f"part.1.{suffix}"];loaded.append(f"hand.{suffix}")
    for target,sources in ((0,(0,)),(1,(1,2)),(2,(1,2)),(3,(3,))):
        for suffix in ("weight","bias"):
            new[f"norms.{target}.{suffix}"]=sum(old[f"part_norm.{s}.{suffix}"] for s in sources)/len(sources);loaded.append(f"norms.{target}.{suffix}")
    partition=json.loads(Path(partition_path).read_text(encoding="utf-8"));one=sorted((r for r in partition["classes"] if r["hand_type"]=="one"),key=lambda r:r["onehand_label"]);indices=torch.tensor([r["label_index"] for r in one])
    new["onehand_classifier.weight"]=old["classifier.weight"][indices].clone();new["onehand_classifier.bias"]=old["classifier.bias"][indices].clone();loaded.extend(["onehand_classifier.weight","onehand_classifier.bias"])
    model.load_state_dict(new);return len(set(loaded))

@torch.no_grad()
def evaluate(model,dl,dev):
    model.eval();n=full_ok=one_n=one_ok=type_ok=0;cos=0
    for b in dl:
        p=b["padding_mask"].to(dev);d=b["detected"].to(dev);y=b["labels"].to(dev);ht=b["hand_types"].to(dev);oy=b["onehand_labels"].to(dev)
        full=model(b["x_full"].to(dev),p,d,b["full_view"].to(dev));part=model(b["x_partial"].to(dev),p,d,b["partial_view"].to(dev));mask=ht==0
        n+=len(y);full_ok+=(full["full_logits"].argmax(1)==y).sum().item();type_ok+=(full["hand_type_logits"].argmax(1)==ht).sum().item();cos+=(full["embedding"]*part["embedding"]).sum().item()
        one_n+=mask.sum().item();one_ok+=(part["onehand_logits"][mask].argmax(1)==oy[mask]).sum().item()
    return {"full_top1":full_ok/n,"onehand_top1":one_ok/max(1,one_n),"hand_type_top1":type_ok/n,"pair_cosine":cos/n}
def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--mode",choices=["overfit32","smoke10","train","resume"],default="train");a=p.parse_args();cfg=load_config(a.config);seed_everything(cfg["experiment"]["seed"]);dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data=resolve(cfg,cfg["paths"]["workspace_data"]);manifest=resolve(cfg,cfg["paths"]["source_manifest"]);tr=HybridDataset(manifest,data/"word_partition_report.json","train",cfg["features"]["max_len"]);va=HybridDataset(manifest,data/"word_partition_report.json","val",cfg["features"]["max_len"],True);epochs=cfg["training"]["max_epochs"]
    if a.mode=="overfit32":
        one=[i for i,r in enumerate(tr.rows) if tr.by_label[int(r["label_index"])]["hand_type"]=="one"][:16]
        two=[i for i,r in enumerate(tr.rows) if tr.by_label[int(r["label_index"])]["hand_type"]=="two"][:16]
        tr=Subset(tr,one+two);va=tr;epochs=200
    elif a.mode=="smoke10":tr=Subset(tr,range(320));va=Subset(va,range(100));epochs=5
    loader=lambda ds,s:DataLoader(ds,batch_size=cfg["training"]["batch_size"],shuffle=s,num_workers=cfg["training"]["num_workers"],collate_fn=collate_hybrid,persistent_workers=cfg["training"]["num_workers"]>0)
    tl,vl=loader(tr,True),loader(va,False);model=make_model(cfg)
    if a.mode in ("train","smoke10"):print(json.dumps({"baseline_parameters_loaded":initialize_from_baseline(model,resolve(cfg,cfg["paths"]["baseline_checkpoint"]),data/"word_partition_report.json")}),flush=True)
    model=model.to(dev);new_names=("onehand_classifier","hand_type_classifier");new_params=[p for n,p in model.named_parameters() if n.startswith(new_names)];backbone_params=[p for n,p in model.named_parameters() if not n.startswith(new_names)];opt=torch.optim.AdamW([{"params":backbone_params,"lr":cfg["training"].get("backbone_learning_rate",cfg["training"]["learning_rate"])},{"params":new_params,"lr":cfg["training"]["learning_rate"]}],weight_decay=cfg["training"]["weight_decay"]);runmode="train" if a.mode=="resume" else ("overfit32_balanced" if a.mode=="overfit32" else a.mode);run=resolve(cfg,cfg["paths"]["runs"])/(cfg["experiment"]["name"]+"_"+runmode);run.mkdir(parents=True,exist_ok=True);last=run/"last.pt";start=0;best=-1.;stale=0
    if a.mode=="resume" and last.exists():ck=torch.load(last,map_location=dev,weights_only=False);model.load_state_dict(ck["model_state"]);opt.load_state_dict(ck["optimizer_state"]);start=ck["epoch"]+1;best=ck["best_metric"]
    fields=["epoch","loss","full_ce","onehand_ce","alignment","supcon","hand_type_ce","full_top1","onehand_top1","hand_type_top1","pair_cosine","seconds"];hist=run/"history.csv"
    with hist.open("a",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        if f.tell()==0:w.writeheader()
        for epoch in range(start,epochs):
            tick=time.time();model.train();sums={k:0. for k in fields[1:7]};seen=0
            for b in tl:
                p=b["padding_mask"].to(dev);d=b["detected"].to(dev);y=b["labels"].to(dev);ht=b["hand_types"].to(dev);oy=b["onehand_labels"].to(dev);opt.zero_grad(set_to_none=True)
                full=model(b["x_full"].to(dev),p,d,b["full_view"].to(dev));part=model(b["x_partial"].to(dev),p,d,b["partial_view"].to(dev));loss,parts=hybrid_loss(full,part,y,ht,oy,cfg["loss"],cfg["loss"]["temperature"]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["training"]["gradient_clip"]);opt.step();n=len(y);seen+=n;sums["loss"]+=loss.item()*n
                for k,v in parts.items():sums[k]+=v.item()*n
            val=evaluate(model,vl,dev);rec={"epoch":epoch,**{k:v/seen for k,v in sums.items()},**val,"seconds":time.time()-tick};w.writerow(rec);f.flush();print(json.dumps(rec),flush=True);score=val["onehand_top1"]+.25*val["full_top1"]
            improved=score>best;best=max(best,score);stale=0 if improved else stale+1;state={"model_state":model.state_dict(),"optimizer_state":opt.state_dict(),"epoch":epoch,"best_metric":best,"config":cfg};torch.save(state,last)
            if improved:torch.save(state,run/"best.pt")
            if a.mode=="overfit32" and val["full_top1"]>=.95 and val["onehand_top1"]>=.95:break
            if a.mode in ("train","resume") and stale>=cfg["training"]["early_stopping_patience"]:break
    dump_json(run/"metrics.json",{"best_selection_metric":best,"device":str(dev)})
if __name__=="__main__":main()
