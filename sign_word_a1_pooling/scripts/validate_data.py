from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import dump_json, load_config, resolve


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
    cfg = load_config(args.config)
    manifest = resolve(cfg, cfg["paths"]["workspace_data"]) / "samples.csv"
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    errors, lengths = [], []
    split_classes = defaultdict(Counter)
    handles = {}
    try:
        for i, row in enumerate(rows, 1):
            path, group_name = row["h5_path"], row["group_name"]
            try:
                handle = handles.setdefault(path, h5py.File(path, "r", swmr=True))
                group = handle[group_name]
                features = group["features"]
                part_mask = group["part_mask"]
                if features.ndim != 2 or features.shape[1] != cfg["data"]["feature_dim"]: raise ValueError("feature shape")
                if features.shape[0] <= 0 or part_mask.shape != (features.shape[0], 4): raise ValueError("length/mask shape")
                if group.attrs.get("feature_version") != cfg["data"]["feature_version"]: raise ValueError("feature_version")
                if int(group.attrs["word_id"]) != int(row["word_id"]): raise ValueError("word_id")
                if str(group.attrs["actor_id"]).zfill(2) != row["actor_id"]: raise ValueError("actor_id")
                array = features[...]
                if not np.isfinite(array).all(): raise ValueError("NaN/Inf")
                frame_index = group["frame_index"][...]
                if not np.array_equal(frame_index, np.arange(len(frame_index))): raise ValueError("frame_index")
                lengths.append(len(features)); split_classes[row["split"]][int(row["label_index"])] += 1
            except Exception as exc:
                errors.append({"video_id": row["video_id"], "error": repr(exc)})
            if i % 1000 == 0: print(f"validated {i}/{len(rows)}", flush=True)
    finally:
        for handle in handles.values(): handle.close()
    missing = {split: sorted(set(range(300)) - set(counts)) for split, counts in split_classes.items()}
    q = np.percentile(lengths, [50, 90, 95, 99]).tolist() if lengths else []
    report = {"samples": len(rows), "valid": len(rows)-len(errors), "errors": len(errors),
              "length": {"min": int(min(lengths)), "median": q[0], "p90": q[1], "p95": q[2],
                         "p99": q[3], "max": int(max(lengths))} if lengths else {},
              "missing_classes_by_split": missing}
    data_dir = manifest.parent
    dump_json(data_dir / "validation_report.json", report)
    dump_json(data_dir / "validation_errors.json", errors)
    print(json.dumps(report, ensure_ascii=False))
    if errors or any(missing.values()): raise SystemExit(1)


if __name__ == "__main__": main()

