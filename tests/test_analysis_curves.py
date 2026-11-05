"""Scripts/analysis/curves.py on the synthetic repo of test_analysis_hypotheses: automation at budgets 1% to 20%, seed means,
Kev and SFT paired as in H3, no-thinking frontier curves only. Synthetic rows only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_analysis_hypotheses import STRONG, WEAK, hy, repo, rows_for, write_run  # noqa: E402

import curves as cv  # noqa: E402  (scripts/analysis is on sys.path via test_analysis_hypotheses)


def test_budgets_and_monotone_curves(tmp_path):
    root, frozen = repo(tmp_path, tasks=("p1",))
    out = cv.build(root, frozen, tasks=("p1",))
    assert out["budgets"][0] == 0.01 and out["budgets"][-1] == 0.20 and len(out["budgets"]) == 20
    for row in ("kev4b-3k", "sft4b-3k", "kev4b-zeroshot", "qwen4b-zeroshot-k0"):
        c = out["rows"][row]["p1"]
        assert len(c) == 20 and all(a <= b + 1e-12 for a, b in zip(c, c[1:]))
    assert out["rows"]["kev4b-full"] == {}            # not frozen in this repo: no curve, no crash


def test_five_percent_point_equals_h3_per_task_automation(tmp_path):
    root, frozen = repo(tmp_path, tasks=("p1",), drop_sft=4)
    out = cv.build(root, frozen, tasks=("p1",))
    h3 = hy.h3(root, frozen, "test", ("p1",), samples=20, seed=1)
    i = out["budgets"].index(0.05)
    assert out["rows"]["kev4b-3k"]["p1"][i] == pytest.approx(h3["per_task"]["p1"]["automation_at_5pct"]["kev"])
    assert out["rows"]["sft4b-3k"]["p1"][i] == pytest.approx(h3["per_task"]["p1"]["automation_at_5pct"]["sft"])


def test_frontier_curves_are_no_thinking_only(tmp_path):
    root, frozen = repo(tmp_path, tasks=("p1",))
    for model in hy.FRONTIER:
        for m in hy.FRONTIER_MODES:
            write_run(root, f"runs/frontier-{model}-{m}-p1", "test", rows_for(tmp_path, STRONG if "nothink" in m else WEAK, seed=7))
    out = cv.build(root, frozen, tasks=("p1",))
    assert set(out["frontier"]) == {f"{model}/{m}/p1" for model in hy.FRONTIER for m in ("nothink-k0", "nothink-k3")}


def test_main_writes_aggregates_only(tmp_path):
    root, frozen = repo(tmp_path, tasks=("p1",))
    cv.main(["--root", str(root), "--out", "c.json"])
    text = (root / "c.json").read_text()
    assert "synthetic" not in text and "r1" not in text   # no row content or record ids
    assert json.loads(text)["split"] == "test"
