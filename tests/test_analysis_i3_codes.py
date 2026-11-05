"""Scripts/analysis/i3_codes.py on synthetic reports: counts reports with an ICD-O code and label agreement; no text out."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("i3_codes", REPO / "scripts" / "analysis" / "i3_codes.py")
ic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ic)


def test_counts_codes_and_agreement(tmp_path):
    rows = [("synthetic adenocarcinoma 8140/3", True), ("synthetic adenoma 8140/0", True), ("no code here", False),
            ("synthetic in situ 8500/2", False)]
    p = tmp_path / "splits" / "i3" / "test.jsonl"
    p.parent.mkdir(parents=True)
    p.write_text("".join(json.dumps({"state": s, "questions": {"malignant": {"label": y}}}) + "\n" for s, y in rows))
    ic.main(["--root", str(tmp_path), "--out", "o.json"])
    text = (tmp_path / "o.json").read_text()
    assert json.loads(text)["test"] == {"n_records": 4, "with_code": 3, "label_agrees": 2}
    assert "synthetic" not in text
