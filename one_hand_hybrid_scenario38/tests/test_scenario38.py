import json
from pathlib import Path


def test_scenario38_mapping():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "data/scenario38_v1/scenario38_report.json").read_text(encoding="utf-8"))
    assert report["total"] == 38
    assert len(report["classes"]) == 38
    assert len({row["word"] for row in report["classes"]}) == 38
    assert len({row["base_label_index"] for row in report["classes"]}) == 38
