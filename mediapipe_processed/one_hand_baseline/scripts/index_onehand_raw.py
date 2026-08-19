from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument("--raw-root",required=True); p.add_argument("--output",required=True); a=p.parse_args()
    rows=[]; rejected=[]
    for path in Path(a.raw_root).rglob("*.json"):
        try:
            raw=path.read_bytes(); obj=json.loads(raw); source=obj.get("source",{}); frames=obj.get("frames",[])
            if not frames: raise ValueError("empty frames")
            video=source.get("video_id") or path.stem; word=source.get("word_id")
            actor=source.get("actor_id"); camera=source.get("camera_id")
            ratios=[]
            for key in ("face","pose","right_hand","left_hand"):
                ratios.append(sum(bool(f.get(key,{}).get("detected",f.get(key) is not None)) for f in frames)/len(frames))
            status="ok"
            rows.append({"path":str(path.resolve()),"video_id":video,"word_id":word,"actor_id":actor,"camera_id":camera,"n_frames":len(frames),"face_ratio":ratios[0],"pose_ratio":ratios[1],"right_ratio":ratios[2],"left_ratio":ratios[3],"status":status,"raw_sha256":hashlib.sha256(raw).hexdigest()})
        except Exception as e: rejected.append({"path":str(path),"error":str(e)})
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    fields=["path","video_id","word_id","actor_id","camera_id","n_frames","face_ratio","pose_ratio","right_ratio","left_ratio","status","raw_sha256"]
    with out.open("w",encoding="utf-8-sig",newline="") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    (out.parent/"rejected_samples.json").write_text(json.dumps(rejected,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"accepted":len(rows),"rejected":len(rejected)}))
    if rejected: sys.exit(2)
if __name__=="__main__": main()
