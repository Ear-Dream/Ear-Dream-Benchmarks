from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.common import load_config,resolve,dump_json
from src.data.onehand_dataset import OneHandDataset,collate_onehand
from src.models.a1p_mask_aware import make_model

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--prototypes",required=True); p.add_argument("--split",default="test"); p.add_argument("--modes",nargs="+",default=["full","right_only","left_only"]); a=p.parse_args(); cfg=load_config(a.config); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=make_model(cfg).to(dev); model.load_state_dict(torch.load(a.checkpoint,map_location=dev,weights_only=False)["model_state"]); model.eval(); proto=np.load(a.prototypes); results={}
    with torch.no_grad():
        for mode in a.modes:
            side="right_only" if mode=="full" else mode; ds=OneHandDataset(resolve(cfg,cfg["paths"]["workspace_data"])/"samples.csv",a.split,cfg["features"]["max_len"],deterministic_side=side); dl=DataLoader(ds,batch_size=cfg["training"]["batch_size"],num_workers=cfg["training"]["num_workers"],collate_fn=collate_onehand)
            bank=torch.from_numpy(proto[{"full":"full","right_only":"right","left_only":"left"}[mode]]).to(dev); ranks=[]; cls_ok=0; total=0
            for b in dl:
                x=b["x_full"] if mode=="full" else b["x_partial"]; v=b["full_view_mask"] if mode=="full" else b["partial_view_mask"]; y=b["labels"].to(dev)
                out=model(x.to(dev),b["padding_mask"].to(dev),b["detected_mask"].to(dev),v.to(dev)); scores=out["embedding"]@bank.T; order=scores.argsort(1,descending=True); ranks.extend(((order==y[:,None]).nonzero()[:,1]+1).cpu().tolist()); cls_ok+=(out["logits"].argmax(1)==y).sum().item(); total+=len(y)
            r=torch.tensor(ranks,dtype=torch.float); results[mode]={"classification_top1":cls_ok/total,"mrr":(1/r).mean().item(),"median_rank":r.median().item(),**{f"recall_at_{k}":(r<=k).float().mean().item() for k in cfg["evaluation"]["candidate_k"]}}
    out=Path(a.checkpoint).parent/f"{a.split}_onehand_metrics.json"; dump_json(out,results); print(json.dumps(results,indent=2))
if __name__=="__main__":main()
