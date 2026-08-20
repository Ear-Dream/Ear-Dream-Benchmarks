from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import dump_json, load_config, resolve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    with resolve(cfg, cfg["paths"]["scenario_csv"]).open(encoding="utf-8-sig", newline="") as f:
        scenario = list(csv.DictReader(f))
    metadata = json.loads(resolve(cfg, cfg["paths"]["class_metadata"]).read_text(encoding="utf-8"))
    partition = json.loads(resolve(cfg, cfg["paths"]["partition_report"]).read_text(encoding="utf-8"))
    by_word = {v["word"]: {"label_index": int(k), "word_id": int(v["word_id"])} for k, v in metadata.items()}
    part_by_label = {int(r["label_index"]): r for r in partition["classes"]}

    classes = []
    for scenario_label, row in enumerate(scenario):
        word = row["단어"].strip()
        if word not in by_word:
            raise ValueError(f"scenario word missing from base 300 model: {word}")
        base = by_word[word]
        part = part_by_label[base["label_index"]]
        classes.append(
            {
                "scenario_label": scenario_label,
                "base_label_index": base["label_index"],
                "word_id": base["word_id"],
                "word": word,
                "hand_type": part["hand_type"],
                "scenario": row["사용 시나리오"],
                "purpose": row["주요 용도"],
            }
        )
    if len(classes) != 38 or len({r["word"] for r in classes}) != 38:
        raise ValueError("scenario list must contain 38 unique words")

    counts = Counter(r["hand_type"] for r in classes)
    output = resolve(cfg, cfg["paths"]["workspace_data"])
    output.mkdir(parents=True, exist_ok=True)
    dump_json(output / "scenario38_report.json", {"total": 38, "hand_type_counts": counts, "classes": classes})
    dump_json(output / "classes.json", [r["word"] for r in classes])
    with (output / "scenario38_classes.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(classes[0]))
        writer.writeheader()
        writer.writerows(classes)
    print(json.dumps({"total": 38, "hand_type_counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
