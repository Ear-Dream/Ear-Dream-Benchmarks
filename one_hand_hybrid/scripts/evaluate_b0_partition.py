from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import load_config,resolve,dump_json
from src.data.hybrid_dataset import HybridDataset,collate_hybrid
from src.models.baseline_model import MaskAwareCandidateModel

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);a=p.parse_args();cfg=load_config(a.config);dev=torch.device("cuda" if torch.cuda.is_available() else "cpu");ckpath=resolve(cfg,cfg["paths"]["baseline_checkpoint"]);ck=torch.load(ckpath,map_location=dev,weights_only=False);old=ck["config"]["model"]
    model=MaskAwareCandidateModel(old["num_classes"],old["d_model"],old["encoder_layers"],old["attention_heads"],old["ffn_dim"],old["conv_kernel_size"],old["dropout"],old["candidate_embedding_dim"],ck["config"]["features"]["max_len"]).to(dev);model.load_state_dict(ck["model_state"]);model.eval();proto=np.load(ckpath.parent/"prototypes.npz")
    ds=HybridDataset(resolve(cfg,cfg["paths"]["source_manifest"]),resolve(cfg,cfg["paths"]["workspace_data"])/"word_partition_report.json","test",cfg["features"]["max_len"],True);dl=DataLoader(ds,batch_size=cfg["training"]["batch_size"],num_workers=cfg["training"]["num_workers"],collate_fn=collate_hybrid);result={}
    with torch.no_grad():
        for mode,key in (("full","full"),("right_only","right"),("left_only","left")):
            bank=torch.from_numpy(proto[key]).to(dev);groups={"one":{"ranks":[],"correct":0,"n":0},"two":{"ranks":[],"correct":0,"n":0}}
            for b in dl:
                if mode=="full":x=b["x_full"];view=b["full_view"]
                else:
                    x=b["x_full"].clone();view=torch.zeros_like(b["full_view"]);side=0 if mode=="right_only" else 1;view[...,side]=1
                    if side==0:x[...,92:134]=0
                    else:x[...,50:92]=0
                y=b["labels"].to(dev);out=model(x.to(dev),b["padding_mask"].to(dev),b["detected"].to(dev),view.to(dev));order=(out["embedding"]@bank.T).argsort(1,descending=True);ranks=((order==y[:,None]).nonzero()[:,1]+1).cpu();pred=out["logits"].argmax(1).cpu()
                for i,ht in enumerate(b["hand_types"]):
                    g="one" if int(ht)==0 else "two";groups[g]["ranks"].append(int(ranks[i]));groups[g]["correct"]+=int(pred[i]==b["labels"][i]);groups[g]["n"]+=1
            result[mode]={}
            for g,v in groups.items():
                r=torch.tensor(v["ranks"]);result[mode][g]={"n":v["n"],"classification_top1":v["correct"]/v["n"],"mrr":float((1/r).mean()),"recall_at_1":float((r<=1).float().mean()),"recall_at_5":float((r<=5).float().mean()),"recall_at_10":float((r<=10).float().mean())}
    out=resolve(cfg,cfg["paths"]["workspace_data"])/"b0_partition_metrics.json";dump_json(out,result);print(json.dumps(result,indent=2))
if __name__=="__main__":main()
