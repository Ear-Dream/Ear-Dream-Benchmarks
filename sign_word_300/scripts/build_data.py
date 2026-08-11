from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import dump_json, load_config, resolve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_dir = resolve(cfg, cfg["paths"]["workspace_data"])
    data_dir.mkdir(parents=True, exist_ok=True)

    selected_path = resolve(cfg, cfg["paths"]["selected_words"])
    with selected_path.open("r", encoding="utf-8-sig", newline="") as stream:
        selected = list(csv.DictReader(stream))
    if len(selected) != 300:
        raise ValueError(f"Expected 300 selected words, found {len(selected)}")
    word_ids = [int(r["원본 단어 번호"]) for r in selected]
    if len(set(word_ids)) != 300:
        raise ValueError("Selected word IDs are not unique")
    classes = {f"WORD{word_id:04d}": i for i, word_id in enumerate(word_ids)}
    names = {str(i): {"word_id": word_ids[i], "word": selected[i]["단어"]} for i in range(300)}

    actor_split = {}
    for split in ("train", "val", "test"):
        for actor in cfg["data"][f"{split}_actors"]:
            if actor in actor_split:
                raise ValueError(f"Actor {actor} appears in multiple splits")
            actor_split[actor] = split

    source_manifest = resolve(cfg, cfg["paths"]["source_manifest"])
    h5_root = resolve(cfg, cfg["paths"]["source_h5_root"])
    with source_manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    chosen = []
    allowed = set(word_ids)
    for row in source_rows:
        word_id = int(row["word_id"])
        actor = str(row["actor_id"]).zfill(2)
        if word_id not in allowed or actor not in actor_split:
            continue
        chosen.append({
            "h5_path": str((h5_root / row["shard"]).resolve()),
            "group_name": row["video_id"], "video_id": row["video_id"],
            "word_id": word_id, "word": selected[word_ids.index(word_id)]["단어"],
            "label_index": classes[f"WORD{word_id:04d}"], "actor_id": actor,
            "camera_id": row["camera_id"], "num_frames": int(row["frames"]),
            "feature_dim": int(row["feature_dim"]), "split": actor_split[actor],
            "pose_detection_rate": row["pose_detection_rate"],
            "right_hand_detection_rate": row["right_hand_detection_rate"],
            "left_hand_detection_rate": row["left_hand_detection_rate"],
            "face_detection_rate": row["face_detection_rate"],
        })
    if not chosen:
        raise RuntimeError("No selected samples found")
    out_manifest = data_dir / "samples.csv"
    with out_manifest.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(chosen[0]))
        writer.writeheader(); writer.writerows(chosen)
    dump_json(data_dir / "classes.json", classes)
    dump_json(data_dir / "class_metadata.json", names)
    dump_json(data_dir / "splits.json", {
        "train_actor_ids": cfg["data"]["train_actors"],
        "val_actor_ids": cfg["data"]["val_actors"],
        "test_actor_ids": cfg["data"]["test_actors"],
        "policy": "actor-independent; all existing cameras for word+actor stay together",
    })
    present={(int(r["word_id"]),r["actor_id"],r["camera_id"]) for r in chosen}
    expected={(word_id,actor,camera) for word_id in word_ids for actor in actor_split for camera in "DFLRU"}
    dump_json(data_dir / "missing_samples.json", [
        {"word_id":word_id,"actor_id":actor,"camera_id":camera}
        for word_id,actor,camera in sorted(expected-present)
    ])
    source_inventory=[]
    for path in sorted({Path(r["h5_path"]) for r in chosen}):
        stat=path.stat(); source_inventory.append({"path":str(path),"size_bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns})
    dump_json(data_dir / "source_h5_inventory.json",source_inventory)
    shutil_target = data_dir / "selected_words.csv"
    shutil_target.write_bytes(selected_path.read_bytes())
    digest = hashlib.sha256(out_manifest.read_bytes()).hexdigest()
    (data_dir / "manifest_hash.txt").write_text(digest + "\n", encoding="ascii")
    print(json.dumps({"samples": len(chosen), "splits": Counter(r["split"] for r in chosen),
                      "classes": len(classes), "manifest_sha256": digest}, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
