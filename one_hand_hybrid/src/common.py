from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
import torch, yaml

def load_config(path):
    path=Path(path).resolve()
    with path.open(encoding="utf-8") as f: cfg=yaml.safe_load(f)
    cfg["_root"]=str(path.parent.parent); cfg["_config_path"]=str(path)
    return cfg

def resolve(cfg,value): return (Path(cfg["_root"])/value).resolve()
def dump_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
