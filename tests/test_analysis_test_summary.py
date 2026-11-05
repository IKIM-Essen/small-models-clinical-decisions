"""Scripts/analysis/test_summary.py on the synthetic repo of test_analysis_hypotheses: [accuracy, macro-F1, automation] per cell,
seed means, no automation for thinking modes, null for rows not run. Synthetic rows only."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_analysis_hypotheses import STRONG, WEAK, hy, repo, rows_for, write_run  # noqa: E402

import test_summary as ts  # noqa: E402


def test_cells_match_the_hypothesis_numbers(tmp_path):
    root, frozen = repo(tmp_path, tasks=("p1",))
    s = ts.rows_summary(root, frozen, tasks=("p1",))
    h3 = hy.h3(root, frozen, "test", ("p1",), samples=20, seed=1)
    acc, f1, auto = s["kev4b-3k"]["p1"]
    assert acc == pytest.approx(h3["per_task"]["p1"]["accuracy"]["kev"])
    assert auto == pytest.approx(h3["per_task"]["p1"]["automation_at_5pct"]["kev"])
    assert s["kev4b-full"]["p1"] is None and s["qwen4b-zeroshot-k3"]["p1"] is None   # not frozen here


def test_thinking_modes_have_no_automation(tmp_path):
    root, _ = repo(tmp_path, tasks=("p1",))
    for model in hy.FRONTIER:
        for m in hy.FRONTIER_MODES:
            write_run(root, f"runs/frontier-{model}-{m}-p1", "test", rows_for(tmp_path, STRONG if "nothink" in m else WEAK, seed=7))
    f = ts.frontier_summary(root, tasks=("p1",))
    assert len(f) == 8
    assert all((v[2] is None) == ("/think" in k) for k, v in f.items())


def test_no_frontier_automation_on_i2(tmp_path):
    root, _ = repo(tmp_path, tasks=("i2",))
    for model in hy.FRONTIER:
        write_run(root, f"runs/frontier-{model}-nothink-k0-i2", "test", rows_for(tmp_path, STRONG, seed=7))
    f = ts.frontier_summary(root, tasks=("i2",))
    assert f and all(v[2] is None and v[0] is not None for v in f.values())
