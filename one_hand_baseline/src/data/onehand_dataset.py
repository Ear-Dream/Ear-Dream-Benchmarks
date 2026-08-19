from __future__ import annotations
import csv
from pathlib import Path
import h5py, numpy as np, torch
from torch.utils.data import Dataset
from .view_generator import make_view

class OneHandDataset(Dataset):
    def __init__(self,manifest,split,max_len=256,paired=True,deterministic_side=None):
        with Path(manifest).open(encoding="utf-8-sig",newline="") as f:
            self.rows=[r for r in csv.DictReader(f) if r["split"]==split]
        self.max_len=max_len; self.paired=paired; self.side=deterministic_side; self._handles={}
    def __len__(self): return len(self.rows)
    def _group(self,row):
        p=row["h5_path"]
        if p not in self._handles: self._handles[p]=h5py.File(p,"r",swmr=True)
        return self._handles[p][row["group_name"]]
    def __getstate__(self): d=self.__dict__.copy(); d["_handles"]={}; return d
    def __getitem__(self,i):
        row=self.rows[i]; g=self._group(row); x=np.asarray(g["features"],dtype=np.float32)
        pm=np.asarray(g["part_mask"],dtype=np.uint8)
        # Existing schema is [pose,right,left,face]. Preserve raw detector state.
        detected=pm[:,1:3]
        if len(x)>self.max_len:
            take=np.linspace(0,len(x)-1,self.max_len).round().astype(np.int64)
            x=x[take]; detected=detected[take]
        x=torch.from_numpy(x); detected=torch.from_numpy(detected)
        mode=self.side or ("right_only" if torch.rand(())<.5 else "left_only")
        partial,view,valid=make_view(x,detected,mode)
        return {"x_full":x,"x_partial":partial,"detected_mask":detected,
                "full_view_mask":torch.ones_like(detected),"partial_view_mask":view,
                "valid_mask":valid,"label":int(row["label_index"]),"mode":mode,
                "video_id":row["video_id"],"actor_id":row["actor_id"],"camera_id":row["camera_id"]}

def collate_onehand(batch):
    n=max(len(v["x_full"]) for v in batch); b=len(batch)
    out={k:torch.zeros(b,n,208) for k in ("x_full","x_partial")}
    for k in ("detected_mask","full_view_mask","partial_view_mask","valid_mask"):
        out[k]=torch.zeros(b,n,2,dtype=torch.uint8)
    out["padding_mask"]=torch.ones(b,n,dtype=torch.bool)
    for i,v in enumerate(batch):
        t=len(v["x_full"]); out["padding_mask"][i,:t]=False
        for k in ("x_full","x_partial","detected_mask","full_view_mask","partial_view_mask","valid_mask"): out[k][i,:t]=v[k]
    out["labels"]=torch.tensor([v["label"] for v in batch]); out["modes"]=[v["mode"] for v in batch]
    out["video_ids"]=[v["video_id"] for v in batch]
    return out
