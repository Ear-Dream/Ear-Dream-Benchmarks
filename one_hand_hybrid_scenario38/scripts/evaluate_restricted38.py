from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import dump_json, load_config, resolve
from src.data.hybrid_dataset import HybridDataset, collate_hybrid
from src.models.hybrid_model import make_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = json.loads((resolve(cfg, cfg["paths"]["workspace_data"]) / "scenario38_report.json").read_text(encoding="utf-8"))
    selected = torch.tensor([r["base_label_index"] for r in report["classes"]], dtype=torch.long, device=dev)
    selected_set = set(selected.cpu().tolist())

    model = make_model(cfg).to(dev)
    checkpoint = torch.load(resolve(cfg, cfg["paths"]["base_checkpoint"]), map_location=dev, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    banks = np.load(resolve(cfg, cfg["paths"]["base_prototypes"]))

    base = HybridDataset(
        resolve(cfg, cfg["paths"]["source_manifest"]),
        resolve(cfg, cfg["paths"]["partition_report"]),
        "test",
        cfg["features"]["max_len"],
        True,
    )
    indices = [i for i, row in enumerate(base.rows) if int(row["label_index"]) in selected_set]
    ds = Subset(base, indices)
    dl = DataLoader(ds, batch_size=cfg["training"]["batch_size"], num_workers=cfg["training"]["num_workers"], collate_fn=collate_hybrid)

    result = {"samples": len(ds), "classes": 38, "device": str(dev), "modes": {}}
    with torch.no_grad():
        for mode in ("full", "right", "left"):
            proto300 = torch.from_numpy(banks[mode]).to(dev)
            proto38 = proto300[selected]
            totals = {g: {"n": 0, "base_cls": 0, "restricted_cls": 0, "base_ret": 0, "restricted_ret": 0, "r5": 0, "r10": 0} for g in ("all", "one", "two")}
            for batch in dl:
                x = batch["x_full"].clone()
                view = batch["full_view"].clone()
                if mode in ("right", "left"):
                    side = 0 if mode == "right" else 1
                    view.zero_(); view[..., side] = 1
                    if side == 0: x[..., 92:134] = 0
                    else: x[..., 50:92] = 0
                output = model(x.to(dev), batch["padding_mask"].to(dev), batch["detected"].to(dev), view.to(dev))
                y = batch["labels"].to(dev)
                base_cls = output["full_logits"].argmax(1)
                restricted_cls = selected[output["full_logits"][:, selected].argmax(1)]
                base_ret = (output["embedding"] @ proto300.T).argmax(1)
                score38 = output["embedding"] @ proto38.T
                order38 = score38.argsort(1, descending=True)
                restricted_ret = selected[order38[:, 0]]
                target38 = torch.stack([(selected == label).nonzero(as_tuple=False)[0, 0] for label in y])
                ranks = (order38 == target38[:, None]).nonzero(as_tuple=False)[:, 1] + 1
                for i in range(len(y)):
                    groups = ("all", "one" if int(batch["hand_types"][i]) == 0 else "two")
                    for group in groups:
                        value = totals[group]; value["n"] += 1
                        value["base_cls"] += int(base_cls[i] == y[i]); value["restricted_cls"] += int(restricted_cls[i] == y[i])
                        value["base_ret"] += int(base_ret[i] == y[i]); value["restricted_ret"] += int(restricted_ret[i] == y[i])
                        value["r5"] += int(ranks[i] <= 5); value["r10"] += int(ranks[i] <= 10)
            result["modes"][mode] = {}
            for group, value in totals.items():
                n = value.pop("n")
                result["modes"][mode][group] = {"n": n, **{key: count / n for key, count in value.items()}}
    output = resolve(cfg, cfg["paths"]["runs"]) / "restricted38_test_metrics.json"
    dump_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
