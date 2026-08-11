from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import dump_json, load_config, resolve
from src.data import SignH5Dataset, collate_sign
from src.model import SPOTER208


def build_model(cfg, classes, checkpoint):
    m = cfg["model"]
    model = SPOTER208(input_dim=m["input_dim"], d_model=m["d_model"], nhead=m["nhead"],
        encoder_layers=m["encoder_layers"], decoder_layers=m["decoder_layers"],
        dim_feedforward=m["dim_feedforward"], dropout=m["dropout"], activation=m["activation"],
        max_sequence_length=cfg["data"]["max_sequence_length"], num_classes=len(classes)).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=False)["model_state"])
    return model


@torch.no_grad()
def collect(model, loader, temperature=1.0):
    logits, labels, videos, actors, cameras, words = [], [], [], [], [], []
    for batch in loader:
        out = model(batch["features"].cuda(non_blocking=True), batch["padding_mask"].cuda(non_blocking=True))
        logits.append((out / temperature).float().cpu()); labels.append(batch["labels"])
        videos += batch["video_ids"]; actors += batch["actor_ids"]; cameras += batch["camera_ids"]; words += batch["word_ids"]
    return torch.cat(logits), torch.cat(labels), videos, actors, cameras, words


def metrics(logits, labels, num_classes):
    probs = logits.softmax(1); pred = probs.argmax(1); cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (labels.numpy(), pred.numpy()), 1)
    per_acc = np.diag(cm) / np.maximum(1, cm.sum(1))
    precision = np.diag(cm) / np.maximum(1, cm.sum(0)); recall = per_acc
    f1 = 2 * precision * recall / np.maximum(1e-12, precision + recall)
    result = {"cross_entropy": float(torch.nn.functional.cross_entropy(logits, labels)),
              "micro_top1": float((pred == labels).float().mean()), "macro_top1": float(per_acc.mean()),
              "top3": float((logits.topk(3, 1).indices == labels[:, None]).any(1).float().mean()),
              "top5": float((logits.topk(5, 1).indices == labels[:, None]).any(1).float().mean()),
              "macro_f1": float(f1.mean()), "mean_confidence": float(probs.max(1).values.mean())}
    return result, cm, per_acc, precision, recall, f1, probs, pred


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args(); cfg=load_config(a.config)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    data=resolve(cfg,cfg["paths"]["workspace_data"]); classes=json.loads((data/"classes.json").read_text(encoding="utf-8")); metadata=json.loads((data/"class_metadata.json").read_text(encoding="utf-8")); n=len(classes)
    run=resolve(cfg,cfg["paths"]["runs"])/(cfg["experiment"]["name"]+"_train"); model=build_model(cfg,classes,run/"best.pt")
    ds=SignH5Dataset(data/"samples.csv","test",cfg["data"]["max_sequence_length"]); loader=DataLoader(ds,batch_size=cfg["training"]["batch_size"],num_workers=cfg["data"]["num_workers"],pin_memory=True,persistent_workers=True,collate_fn=collate_sign)
    calibration=run/"calibration.json"; temperature=json.loads(calibration.read_text())["temperature"] if calibration.exists() else 1.0
    logits,labels,videos,actors,cameras,words=collect(model,loader,temperature); result,cm,acc,prec,rec,f1,probs,pred=metrics(logits,labels,n)
    result["temperature"]=temperature
    by_camera={}
    for camera in sorted(set(cameras)):
        idx=torch.tensor([x==camera for x in cameras]); by_camera[camera]=float((pred[idx]==labels[idx]).float().mean())
    result["accuracy_by_camera"]=by_camera; result["accuracy_by_actor"]={actor:float((pred[torch.tensor([x==actor for x in actors])]==labels[torch.tensor([x==actor for x in actors])]).float().mean()) for actor in sorted(set(actors))}
    dump_json(run/"test_metrics.json",result); np.save(run/"confusion_matrix.npy",cm)
    with (run/"per_class_metrics.csv").open("w",encoding="utf-8-sig",newline="") as s:
        w=csv.writer(s); w.writerow(["label_index","word_id","word","support","accuracy","precision","recall","f1"])
        for i in range(n): w.writerow([i,metadata[str(i)]["word_id"],metadata[str(i)]["word"],int(cm[i].sum()),acc[i],prec[i],rec[i],f1[i]])
    confusions=[]
    for true in range(n):
        for guessed in np.argsort(cm[true])[::-1]:
            if guessed!=true and cm[true,guessed]>0: confusions.append((int(cm[true,guessed]),true,int(guessed)))
    with (run/"top_confusions.csv").open("w",encoding="utf-8-sig",newline="") as s:
        w=csv.writer(s); w.writerow(["count","true_word_id","true_word","predicted_word_id","predicted_word"])
        for count,true,guessed in sorted(confusions,reverse=True)[:100]: w.writerow([count,metadata[str(true)]["word_id"],metadata[str(true)]["word"],metadata[str(guessed)]["word_id"],metadata[str(guessed)]["word"]])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(12,10)); ax.imshow(cm,cmap="Blues",aspect="auto"); ax.set_title("300-word confusion matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("True"); fig.tight_layout(); fig.savefig(run/"confusion_matrix.png",dpi=160); plt.close(fig)
    except Exception as exc:
        result["confusion_plot_warning"] = repr(exc)
        dump_json(run/"test_metrics.json",result)
    print(json.dumps(result,ensure_ascii=False))

if __name__=="__main__": main()
