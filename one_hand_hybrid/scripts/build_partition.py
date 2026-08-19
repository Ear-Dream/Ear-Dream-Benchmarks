from __future__ import annotations
import argparse,csv,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import load_config,resolve,dump_json

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);a=p.parse_args();cfg=load_config(a.config)
    source=resolve(cfg,cfg["paths"]["hand_partition_csv"]);meta=json.loads(resolve(cfg,cfg["paths"]["class_metadata"]).read_text(encoding="utf-8"))
    rows=list(csv.reader(source.read_text(encoding="cp949").splitlines()))[1:];byword=defaultdict(list)
    for r in rows:byword[r[1].strip()].append(r)
    one=[];two=[];classes=[]
    for label_s,item in sorted(meta.items(),key=lambda x:int(x[0])):
        label=int(label_s);vals=list(item.values());word_id=int(vals[0]);word=str(vals[1]).strip();matches=byword[word]
        if not matches:raise ValueError(f"partition missing: {label}/{word_id}/{word}")
        # The selected 300 list contains arm word_id 1147 (CSV selection 208).
        if len(matches)>1 and word_id==1147:match=next(r for r in matches if r[0]=="208")
        elif len(matches)==1:match=matches[0]
        else:raise ValueError(f"ambiguous partition: {label}/{word_id}/{word}")
        hand_type="one" if match[2]=="한손" else "two";bucket=one if hand_type=="one" else two;one_label=len(one) if hand_type=="one" else -1
        row={"label_index":label,"word_id":word_id,"word":word,"hand_type":hand_type,"onehand_label":one_label,"source_selection_index":int(match[0])};classes.append(row);bucket.append(row)
    if (len(one),len(two))!=(106,194):raise ValueError(f"expected 106/194, got {len(one)}/{len(two)}")
    out=resolve(cfg,cfg["paths"]["workspace_data"]);out.mkdir(parents=True,exist_ok=True);dump_json(out/"word_partition_report.json",{"one_hand_classes":len(one),"two_hand_classes":len(two),"classes":classes,"excluded":[{"word":"목요일","reason":"not in selected 300/3000 mapping"},{"selection_index":301,"word":"팔","reason":"duplicate; selected word_id 1147 uses selection 208"}]})
    with (out/"word_partition.csv").open("w",encoding="utf-8-sig",newline="") as f:w=csv.DictWriter(f,fieldnames=list(classes[0]));w.writeheader();w.writerows(classes)
    print(json.dumps({"one_hand":len(one),"two_hand":len(two),"total":len(classes)},ensure_ascii=False))
if __name__=="__main__":main()
