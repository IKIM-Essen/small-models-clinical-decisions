"""Scripts/analysis/seed_spread.py and scripts/analysis/majority.py on synthetic inputs (aggregates only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_analysis_hypotheses import hy, repo  # noqa: E402

import majority as mj  # noqa: E402
import seed_spread as ss  # noqa: E402


def test_seed_spread_means_equal_the_h3_seed_means(tmp_path):
    root, frozen = repo(tmp_path, tasks=("p1",))
    s = ss.spread(root, frozen, tasks=("p1",))
    h3 = hy.h3(root, frozen, "test", ("p1",), samples=20, seed=1)
    for side in ("kev", "sft"):
        v = s["p1"][side]["automation_at_5pct"]
        assert len(v["seeds"]) == 3 and v["min"] <= v["max"] and v["sd"] >= 0
        assert sum(v["seeds"]) / 3 == pytest.approx(h3["per_task"]["p1"]["automation_at_5pct"][side])


def test_majority_uses_the_training_mode_on_test_counts():
    build = {"splits": {"train": {"questions": {"q": {"labels": {"a": 7, "b": 3}}}},
                        "test": {"questions": {"q": {"labels": {"a": 2, "b": 6}}}}}}
    r = mj.majority(build)
    assert r["accuracy"] == 0.25 and r["n_rows"] == 8
    assert r["macro_f1"] == pytest.approx((2 * 2 / (8 + 2)) / 2)   # class a: F1 0.4; class b: 0
