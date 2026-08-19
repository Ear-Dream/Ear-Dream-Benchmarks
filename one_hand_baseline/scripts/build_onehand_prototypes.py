from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.common import load_config,resolve,dump_json
from src.data.onehand_dataset import OneHandDataset,collate_onehand
from src.models.a1p_mask_aware import make_model
from src.retrieval.prototype_bank import build_prototypes

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--output"); a=p.parse_args(); cfg=load_config(a.config); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=make_model(cfg).to(device); ck=torch.load(a.checkpoint,map_location=device,weights_only=False); model.load_state_dict(ck["model_state"]); model.eval(); manifest=resolve(cfg,cfg["paths"]["workspace_data"])/"samples.csv"
    banks={}; counts={}
    with torch.no_grad():
        for mode in ("full","right_only","left_only"):
            ds=OneHandDataset(manifest,"train",cfg["features"]["max_len"],deterministic_side="right_only" if mode=="full" else mode); loader=DataLoader(ds,batch_size=cfg["training"]["batch_size"],num_workers=cfg["training"]["num_workers"],collate_fn=collate_onehand)
            zs=[]; ys=[]
            for b in loader:
                x=b["x_full"] if mode=="full" else b["x_partial"]; v=b["full_view_mask"] if mode=="full" else b["partial_view_mask"]
                out=model(x.to(device),b["padding_mask"].to(device),b["detected_mask"].to(device),v.to(device)); zs.append(out["embedding"].cpu()); ys.append(b["labels"])
            bank,count=build_prototypes(torch.cat(zs),torch.cat(ys),cfg["model"]["num_classes"]); banks[mode]=bank.numpy(); counts[mode]=count.numpy()
    out=Path(a.output) if a.output else Path(a.checkpoint).parent/"prototypes.npz"; np.savez(out,full=banks["full"],right=banks["right_only"],left=banks["left_only"],counts=np.stack([counts["full"],counts["right_only"],counts["left_only"]],1))
    dump_json(out.with_name("prototype_metadata.json"),{"schema":"onehand_class_prototypes_v1","checkpoint":str(Path(a.checkpoint).resolve()),"embedding_dim":cfg["model"]["candidate_embedding_dim"],"split":"train_only"}); print(out)
if __name__=="__main__":main()
