from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.common import load_config,resolve,dump_json
from src.data.onehand_dataset import OneHandDataset,collate_onehand
from src.models.a1p_mask_aware import make_model

def select_threshold(scores,correct,target_precision):
    thresholds=np.unique(np.quantile(scores,np.linspace(0,1,101)))
    curve=[]
    for threshold in thresholds:
        keep=scores>=threshold
        if not keep.any(): continue
        curve.append({"threshold":float(threshold),"coverage":float(keep.mean()),
                      "precision":float(correct[keep].mean()),"accepted":int(keep.sum())})
    eligible=[row for row in curve if row["precision"]>=target_precision]
    selected=max(eligible,key=lambda row:row["coverage"]) if eligible else max(curve,key=lambda row:row["precision"])
    return selected,curve

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--checkpoint",required=True)
    p.add_argument("--prototypes",required=True); p.add_argument("--target-precision",type=float,default=.95); p.add_argument("--output"); a=p.parse_args()
    cfg=load_config(a.config); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=make_model(cfg).to(dev)
    model.load_state_dict(torch.load(a.checkpoint,map_location=dev,weights_only=False)["model_state"]); model.eval(); banks=np.load(a.prototypes); result={"split":"validation","target_precision":a.target_precision,"modes":{}}
    with torch.no_grad():
        for mode,key in (("right_only","right"),("left_only","left")):
            ds=OneHandDataset(resolve(cfg,cfg["paths"]["workspace_data"])/"samples.csv","val",cfg["features"]["max_len"],deterministic_side=mode)
            dl=DataLoader(ds,batch_size=cfg["training"]["batch_size"],num_workers=cfg["training"]["num_workers"],collate_fn=collate_onehand)
            bank=torch.from_numpy(banks[key]).to(dev); maxima=[]; correct=[]
            for b in dl:
                out=model(b["x_partial"].to(dev),b["padding_mask"].to(dev),b["detected_mask"].to(dev),b["partial_view_mask"].to(dev)); score=out["embedding"]@bank.T; top=score.argmax(1); maxima.extend(score.max(1).values.cpu().tolist()); correct.extend((top.cpu()==b["labels"]).tolist())
            selected,curve=select_threshold(np.asarray(maxima),np.asarray(correct,dtype=bool),a.target_precision); result["modes"][mode]={"selected":selected,"curve":curve}
    output=Path(a.output) if a.output else Path(a.checkpoint).parent/"calibration.json"; dump_json(output,result); print(json.dumps({k:v["selected"] for k,v in result["modes"].items()},indent=2))
if __name__=="__main__":main()
