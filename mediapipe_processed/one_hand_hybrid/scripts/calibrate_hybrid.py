from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import load_config,resolve,dump_json
from src.data.hybrid_dataset import HybridDataset,collate_hybrid
from src.models.hybrid_model import make_model
def choose(scores,correct,target=.95):
    curve=[]
    for t in np.unique(np.quantile(scores,np.linspace(0,1,101))):
        keep=scores>=t
        if keep.any():curve.append({"threshold":float(t),"coverage":float(keep.mean()),"precision":float(correct[keep].mean())})
    valid=[r for r in curve if r["precision"]>=target];return (max(valid,key=lambda r:r["coverage"]) if valid else max(curve,key=lambda r:r["precision"])),curve
def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--prototypes",required=True);a=p.parse_args();cfg=load_config(a.config);dev=torch.device("cuda" if torch.cuda.is_available() else "cpu");m=make_model(cfg).to(dev);m.load_state_dict(torch.load(a.checkpoint,map_location=dev,weights_only=False)["model_state"]);m.eval();banks=np.load(a.prototypes);ds=HybridDataset(resolve(cfg,cfg["paths"]["source_manifest"]),resolve(cfg,cfg["paths"]["workspace_data"])/"word_partition_report.json","val",cfg["features"]["max_len"],True);dl=DataLoader(ds,batch_size=cfg["training"]["batch_size"],num_workers=cfg["training"]["num_workers"],collate_fn=collate_hybrid);out={"target_precision":.95,"modes":{}}
    with torch.no_grad():
        for mode in ("right","left"):
            proto=torch.from_numpy(banks[mode]).to(dev);scores=[];correct=[]
            for b in dl:
                side=0 if mode=="right" else 1;x=b["x_full"].clone();view=torch.zeros_like(b["full_view"]);view[...,side]=1
                if side==0:x[...,92:134]=0
                else:x[...,50:92]=0
                o=m(x.to(dev),b["padding_mask"].to(dev),b["detected"].to(dev),view.to(dev));s=o["embedding"]@proto.T;scores.extend(s.max(1).values.cpu());correct.extend((s.argmax(1).cpu()==b["labels"]).tolist())
            selected,curve=choose(np.asarray(scores),np.asarray(correct,dtype=bool));out["modes"][mode]={"selected":selected,"curve":curve}
    path=Path(a.checkpoint).parent/"hybrid_calibration.json";dump_json(path,out);print(json.dumps({k:v["selected"] for k,v in out["modes"].items()},indent=2))
if __name__=="__main__":main()
