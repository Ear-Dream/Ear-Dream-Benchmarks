from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import load_config,resolve,dump_json
from src.data.hybrid_dataset import HybridDataset,collate_hybrid
from src.models.hybrid_model import make_model

def bank(z,y,n=300):
    out=torch.zeros(n,z.shape[1]);count=torch.zeros(n);out.index_add_(0,y,z);count.index_add_(0,y,torch.ones(len(y)));return torch.nn.functional.normalize(out/count.clamp_min(1)[:,None],dim=-1),count
def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--checkpoint",required=True);a=p.parse_args();cfg=load_config(a.config);dev=torch.device("cuda" if torch.cuda.is_available() else "cpu");model=make_model(cfg).to(dev);model.load_state_dict(torch.load(a.checkpoint,map_location=dev,weights_only=False)["model_state"]);model.eval();ds=HybridDataset(resolve(cfg,cfg["paths"]["source_manifest"]),resolve(cfg,cfg["paths"]["workspace_data"])/"word_partition_report.json","train",cfg["features"]["max_len"],True);dl=DataLoader(ds,batch_size=cfg["training"]["batch_size"],num_workers=cfg["training"]["num_workers"],collate_fn=collate_hybrid);result={};counts={}
    with torch.no_grad():
        for mode in ("full","right","left","selected"):
            zs=[];ys=[]
            for b in dl:
                x=b["x_full"].clone();view=b["full_view"].clone()
                if mode in ("right","left"):
                    side=0 if mode=="right" else 1;view.zero_();view[...,side]=1
                    if side==0:x[...,92:134]=0
                    else:x[...,50:92]=0
                elif mode=="selected":x=b["x_partial"];view=b["partial_view"]
                o=model(x.to(dev),b["padding_mask"].to(dev),b["detected"].to(dev),view.to(dev));zs.append(o["embedding"].cpu());ys.append(b["labels"])
            proto,count=bank(torch.cat(zs),torch.cat(ys));result[mode]=proto.numpy();counts[mode]=count.numpy()
    out=Path(a.checkpoint).parent/"h2_prototypes.npz";np.savez(out,**result,counts=np.stack([counts[k] for k in ("full","right","left","selected")],1));dump_json(out.with_name("h2_prototype_metadata.json"),{"split":"train_only","modes":["full","right","left","selected"],"checkpoint":str(Path(a.checkpoint).resolve())});print(out)
if __name__=="__main__":main()
