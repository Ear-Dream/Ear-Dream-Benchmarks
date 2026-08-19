from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import h5py,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.common import load_config,resolve,dump_json
from src.data.view_generator import make_view

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--limit",type=int); a=p.parse_args(); cfg=load_config(a.config)
    manifest=resolve(cfg,cfg["paths"]["workspace_data"])/"samples.csv"
    with manifest.open(encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
    errors=[]; checked=0; actors={}
    for r in rows[:a.limit]:
        try:
            actor=r["actor_id"]; split=r["split"]
            if actor in actors and actors[actor]!=split: raise ValueError("actor split leakage")
            actors[actor]=split
            with h5py.File(r["h5_path"],"r") as h: g=h[r["group_name"]]; x=np.asarray(g["features"]); pm=np.asarray(g["part_mask"])
            if x.ndim!=2 or x.shape[1]!=208 or pm.shape!=(len(x),4): raise ValueError(f"bad shape {x.shape}/{pm.shape}")
            if not np.isfinite(x).all(): raise ValueError("NaN/Inf")
            xt=torch.from_numpy(x[:min(8,len(x))]); d=torch.from_numpy(pm[:len(xt),1:3]); right,_,_=make_view(xt,d,"right_only"); left,_,_=make_view(xt,d,"left_only")
            if torch.count_nonzero(right[:,92:134]) or torch.count_nonzero(left[:,50:92]): raise ValueError("part masking failed")
            checked+=1
        except Exception as e: errors.append({"video_id":r.get("video_id"),"error":str(e)})
    report={"checked":checked,"errors":len(errors),"passed":not errors}
    dump_json(manifest.parent/"validation_report.json",report); dump_json(manifest.parent/"validation_errors.json",errors); print(json.dumps(report))
    if errors: sys.exit(1)
if __name__=="__main__": main()
