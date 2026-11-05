"""Scripts/analysis/group_overlap.py on synthetic split files: shared group counts, the 3k subset, no ids in the output."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("group_overlap", REPO / "scripts" / "analysis" / "group_overlap.py")
go = importlib.util.module_from_spec(spec)
spec.loader.exec_module(go)


def write(path, gs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"_meta": {"id": f"r{i}", "group": g}, "state": "synthetic"}) + "\n" for i, g in enumerate(gs)))


def test_counts_shared_groups_and_writes_no_ids(tmp_path):
    write(tmp_path / "splits/i3/train.jsonl", ["pa", "pb", "pc", "pc"])
    write(tmp_path / "splits/i3/dev.jsonl", ["pc", "pd"])
    write(tmp_path / "splits/i3/test.jsonl", ["pa", "pd", "pe"])
    write(tmp_path / "data/derived/i3-3k/train.jsonl", ["pa", "pc"])
    go.main(["--root", str(tmp_path), "--out", "o.json"])
    text = (tmp_path / "o.json").read_text()
    r = json.loads(text)["i3"]
    assert r["train_test"] == 1 and r["train_dev"] == 1 and r["dev_test"] == 1
    assert r["test_subset"] == 1 and r["dev_subset"] == 1 and r["n_train"] == 3
    assert "pa" not in text and "synthetic" not in text


def test_tasks_without_split_files_are_skipped(tmp_path):
    go.main(["--root", str(tmp_path), "--out", "o.json"])
    assert json.loads((tmp_path / "o.json").read_text()) == {}
