from __future__ import annotations
import argparse,csv,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.common import load_config,resolve,dump_json

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args(); cfg=load_config(a.config)
    source=resolve(cfg,cfg["paths"]["source_manifest"]); target=resolve(cfg,cfg["paths"]["workspace_data"]); target.mkdir(parents=True,exist_ok=True)
    with source.open(encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
    required={"h5_path","group_name","video_id","label_index","actor_id","split"}
    if not rows or not required.issubset(rows[0]): raise ValueError(f"source manifest must contain {sorted(required)}")
    seen={}; actor_split={}
    for r in rows:
        if r["video_id"] in seen: raise ValueError(f"duplicate video_id: {r['video_id']}")
        seen[r["video_id"]]=1; actor=r["actor_id"]; split=r["split"]
        if actor in actor_split and actor_split[actor]!=split: raise ValueError(f"split leakage for actor {actor}")
        actor_split[actor]=split; r["hand_mask_source"]="part_mask[:,1:3]"
        r["feature_version"]=cfg["features"]["base_version"]; r["schema_version"]=cfg["features"]["schema_version"]
    out=target/"samples.csv"; fields=list(rows[0])
    with out.open("w",encoding="utf-8-sig",newline="") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    for name in ("classes.json","class_metadata.json","splits.json"):
        src=source.parent/name
        if src.exists(): shutil.copy2(src,target/name)
    digest=hashlib.sha256(out.read_bytes()).hexdigest(); (target/"manifest_hash.txt").write_text(digest+"\n",encoding="ascii")
    dump_json(target/"build_manifest.json",{"source_manifest":str(source),"samples":len(rows),"feature_version":cfg["features"]["base_version"],"schema_version":cfg["features"]["schema_version"],"manifest_sha256":digest})
    print(json.dumps({"samples":len(rows),"manifest_sha256":digest}))
if __name__=="__main__": main()
