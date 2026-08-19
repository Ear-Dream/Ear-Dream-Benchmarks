from __future__ import annotations
import csv,json
from pathlib import Path
import h5py,numpy as np,torch
from torch.utils.data import Dataset
from .view_generator import make_view

def motion_side(features,detected):
    if len(features)<2: return 0
    energies=[]
    for a,b,mask in ((50,92,detected[:,0]),(92,134,detected[:,1])):
        velocity=np.abs(np.diff(features[:,a:b],axis=0)).mean(1); valid=mask[1:]>0
        energies.append(float(velocity[valid].mean()) if valid.any() else 0.0)
    return int(energies[1]>energies[0]) # 0 right, 1 left

class HybridDataset(Dataset):
    def __init__(self,manifest,partition,split,max_len=256,deterministic=False):
        with Path(manifest).open(encoding="utf-8-sig",newline="") as f: self.rows=[r for r in csv.DictReader(f) if r["split"]==split]
        report=json.loads(Path(partition).read_text(encoding="utf-8")); self.by_label={int(r["label_index"]):r for r in report["classes"]}
        self.max_len=max_len; self.deterministic=deterministic; self._handles={}
    def __len__(self): return len(self.rows)
    def _group(self,r):
        p=r["h5_path"]
        if p not in self._handles:self._handles[p]=h5py.File(p,"r",swmr=True)
        return self._handles[p][r["group_name"]]
    def __getstate__(self): d=self.__dict__.copy();d["_handles"]={};return d
    def __getitem__(self,i):
        r=self.rows[i];g=self._group(r);x=np.asarray(g["features"],np.float32);pm=np.asarray(g["part_mask"],np.uint8);d=pm[:,1:3]
        if len(x)>self.max_len:
            take=np.linspace(0,len(x)-1,self.max_len).round().astype(np.int64);x=x[take];d=d[take]
        label=int(r["label_index"]); part=self.by_label[label]; selected=motion_side(x,d)
        # One-hand CE always uses its motion-selected hand. Retrieval alternates sides.
        if part["hand_type"]=="one": partial_side=selected
        elif self.deterministic: partial_side=i%2
        else: partial_side=int(torch.rand(())>=.5)
        mode="right_only" if partial_side==0 else "left_only"; xt=torch.from_numpy(x);dt=torch.from_numpy(d);xp,view,valid=make_view(xt,dt,mode)
        return {"x_full":xt,"x_partial":xp,"detected":dt,"full_view":torch.ones_like(dt),"partial_view":view,"valid":valid,
                "label":label,"hand_type":0 if part["hand_type"]=="one" else 1,"onehand_label":int(part["onehand_label"]),
                "selected_side":selected,"mode":mode,"video_id":r["video_id"]}

def collate_hybrid(batch):
    n=max(len(v["x_full"]) for v in batch);b=len(batch);out={k:torch.zeros(b,n,208) for k in ("x_full","x_partial")}
    for k in ("detected","full_view","partial_view","valid"):out[k]=torch.zeros(b,n,2,dtype=torch.uint8)
    out["padding_mask"]=torch.ones(b,n,dtype=torch.bool)
    for i,v in enumerate(batch):
        t=len(v["x_full"]);out["padding_mask"][i,:t]=False
        for k in ("x_full","x_partial","detected","full_view","partial_view","valid"):out[k][i,:t]=v[k]
    for k in ("label","hand_type","onehand_label","selected_side"):out[k+"s"]=torch.tensor([v[k] for v in batch])
    out["video_ids"]=[v["video_id"] for v in batch];out["modes"]=[v["mode"] for v in batch];return out
