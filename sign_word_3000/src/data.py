from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class SignH5Dataset(Dataset):
    def __init__(self, manifest: str | Path, split: str, max_len: int = 256, labels=None):
        with Path(manifest).open("r", encoding="utf-8-sig", newline="") as stream:
            self.rows = [r for r in csv.DictReader(stream) if r["split"] == split]
        if labels is not None:
            allowed = set(labels)
            self.rows = [r for r in self.rows if int(r["label_index"]) in allowed]
        self.max_len = max_len
        self._handles: dict[str, h5py.File] = {}

    def __len__(self):
        return len(self.rows)

    def _handle(self, path: str):
        handle = self._handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r", swmr=True)
            self._handles[path] = handle
        return handle

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __del__(self):
        for handle in getattr(self, "_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass

    def __getitem__(self, index):
        row = self.rows[index]
        group = self._handle(row["h5_path"])[row["group_name"]]
        features = np.asarray(group["features"], dtype=np.float32)
        part_mask = np.asarray(group["part_mask"], dtype=np.uint8)
        if len(features) > self.max_len:
            take = np.linspace(0, len(features) - 1, self.max_len).round().astype(np.int64)
            features, part_mask = features[take], part_mask[take]
        return {
            "features": torch.from_numpy(features),
            "part_mask": torch.from_numpy(part_mask),
            "label": int(row["label_index"]),
            "video_id": row["video_id"],
            "actor_id": row["actor_id"],
            "camera_id": row["camera_id"],
            "word_id": int(row["word_id"]),
        }


def collate_sign(batch):
    lengths = torch.tensor([len(x["features"]) for x in batch], dtype=torch.long)
    length = int(lengths.max())
    features = torch.zeros(len(batch), length, batch[0]["features"].shape[1], dtype=torch.float32)
    part_mask = torch.zeros(len(batch), length, 4, dtype=torch.uint8)
    padding_mask = torch.ones(len(batch), length, dtype=torch.bool)
    for i, item in enumerate(batch):
        n = len(item["features"])
        features[i, :n] = item["features"]
        part_mask[i, :n] = item["part_mask"]
        padding_mask[i, :n] = False
    return {
        "features": features,
        "part_mask": part_mask,
        "padding_mask": padding_mask,
        "labels": torch.tensor([x["label"] for x in batch], dtype=torch.long),
        "lengths": lengths,
        "video_ids": [x["video_id"] for x in batch],
        "actor_ids": [x["actor_id"] for x in batch],
        "camera_ids": [x["camera_id"] for x in batch],
        "word_ids": [x["word_id"] for x in batch],
    }
