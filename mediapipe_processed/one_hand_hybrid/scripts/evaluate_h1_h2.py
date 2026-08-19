from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import load_config,resolve,dump_json
from src.data.hybrid_dataset import HybridDataset,collate_hybrid
from src.models.hybrid_model import make_model

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--prototypes",required=True);a=p.parse_args();cfg=load_config(a.config);dev=torch.device("cuda" if torch.cuda.is_available() else "cpu");model=make_model(cfg).to(dev);model.load_state_dict(torch.load(a.checkpoint,map_location=dev,weights_only=False)["model_state"]);model.eval();banks=np.load(a.prototypes);ds=HybridDataset(resolve(cfg,cfg["paths"]["source_manifest"]),resolve(cfg,cfg["paths"]["workspace_data"])/"word_partition_report.json","test",cfg["features"]["max_len"],True);dl=DataLoader(ds,batch_size=cfg["training"]["batch_size"],num_workers=cfg["training"]["num_workers"],collate_fn=collate_hybrid);result={}
    partition=json.loads((resolve(cfg,cfg["paths"]["workspace_data"])/"word_partition_report.json").read_text(encoding="utf-8"));one_global=torch.tensor([r["label_index"] for r in sorted((r for r in partition["classes"] if r["hand_type"]=="one"),key=lambda r:r["onehand_label"])],device=dev)
    with torch.no_grad():
        for mode in ("full","right","left"):
            proto=torch.from_numpy(banks[mode]).to(dev);groups={g:{"n":0,"full_ok":0,"one_ok":0,"restricted_ok":0,"type_ok":0,"ranks":[]} for g in ("one","two")}
            for b in dl:
                x=b["x_full"].clone();view=b["full_view"].clone()
                if mode in ("right","left"):
                    side=0 if mode=="right" else 1;view.zero_();view[...,side]=1
                    if side==0:x[...,92:134]=0
                    else:x[...,50:92]=0
                o=model(x.to(dev),b["padding_mask"].to(dev),b["detected"].to(dev),view.to(dev));y=b["labels"].to(dev);order=(o["embedding"]@proto.T).argsort(1,descending=True);ranks=((order==y[:,None]).nonzero()[:,1]+1).cpu()
                # Dedicated one-hand result uses motion-selected view, independent of eval mode.
                selected=model(b["x_partial"].to(dev),b["padding_mask"].to(dev),b["detected"].to(dev),b["partial_view"].to(dev))
                for i,ht in enumerate(b["hand_types"]):
                    g="one" if int(ht)==0 else "two";v=groups[g];v["n"]+=1;v["full_ok"]+=int(o["full_logits"][i].argmax().cpu()==b["labels"][i]);v["type_ok"]+=int(o["hand_type_logits"][i].argmax().cpu()==ht);v["ranks"].append(int(ranks[i]))
                    if g=="one":
                        v["one_ok"]+=int(selected["onehand_logits"][i].argmax().cpu()==b["onehand_labels"][i]);restricted=one_global[selected["full_logits"][i,one_global].argmax()]
                        v["restricted_ok"]+=int(restricted.cpu()==b["labels"][i])
            result[mode]={}
            for g,v in groups.items():
                r=torch.tensor(v["ranks"]);result[mode][g]={"n":v["n"],"full_top1":v["full_ok"]/v["n"],"onehand_head_top1":v["one_ok"]/v["n"] if g=="one" else None,"restricted_106_top1":v["restricted_ok"]/v["n"] if g=="one" else None,"hand_type_top1":v["type_ok"]/v["n"],"mrr":float((1/r).mean()),"recall_at_1":float((r<=1).float().mean()),"recall_at_5":float((r<=5).float().mean()),"recall_at_10":float((r<=10).float().mean())}
    out=Path(a.checkpoint).parent/"h1_h2_test_metrics.json";dump_json(out,result);print(json.dumps(result,indent=2))
if __name__=="__main__":main()
