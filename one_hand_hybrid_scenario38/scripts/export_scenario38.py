from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import dump_json, load_config, resolve
from src.models.hybrid_model import make_model


class Scenario38Wrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, indices: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("indices", indices)

    def forward(self, x, padding, detected, view):
        output = self.model(x, padding, detected, view)
        return output["full_logits"].index_select(1, self.indices), output["hand_type_logits"], output["embedding"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = resolve(cfg, cfg["paths"]["workspace_data"])
    report = json.loads((data / "scenario38_report.json").read_text(encoding="utf-8"))
    indices = torch.tensor([r["base_label_index"] for r in report["classes"]], dtype=torch.long)

    model = make_model(cfg)
    state = torch.load(resolve(cfg, cfg["paths"]["base_checkpoint"]), map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    wrapper = Scenario38Wrapper(model, indices).eval()
    example = (torch.zeros(1, 16, 208), torch.zeros(1, 16, dtype=torch.bool), torch.ones(1, 16, 2), torch.ones(1, 16, 2))
    traced = torch.jit.trace(wrapper, example, check_trace=False)

    # Keep the TorchScript save path relative on Windows; libtorch may reject
    # an otherwise valid absolute path when a parent directory contains Hangul.
    output_dir = Path(cfg["paths"]["runs"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "scenario38_model_torchscript.pt"
    traced.save(str(model_path))

    source_banks = np.load(resolve(cfg, cfg["paths"]["base_prototypes"]))
    payload = {mode: source_banks[mode][indices.numpy()] for mode in ("full", "right", "left", "selected")}
    if "counts" in source_banks:
        payload["counts"] = source_banks["counts"][indices.numpy()]
    np.savez(output_dir / "scenario38_prototypes.npz", **payload)

    diffs = {}
    with torch.no_grad():
        for length in (16, 31):
            sample = (torch.randn(2, length, 208), torch.zeros(2, length, dtype=torch.bool), torch.ones(2, length, 2), torch.ones(2, length, 2))
            eager = wrapper(*sample); scripted = traced(*sample)
            diffs[str(length)] = max((a - b).abs().max().item() for a, b in zip(eager, scripted))
    dump_json(output_dir / "scenario38_export_parity.json", {"max_abs_diff": max(diffs.values()), "by_sequence_length": diffs})
    print(json.dumps({"model": str(model_path), "prototypes": str(output_dir / 'scenario38_prototypes.npz'), "max_abs_diff": max(diffs.values())}))


if __name__ == "__main__":
    main()
