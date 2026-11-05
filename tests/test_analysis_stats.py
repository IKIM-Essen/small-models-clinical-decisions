"""Scripts/analysis/stats.py: the statistics protocol. Synthetic rows built through kev's own
row builder (kev.benchmark.prediction_rows), so the schema is exactly what kev.benchmark writes."""
from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "stats.py"


@pytest.fixture(scope="module")
def st():
    spec = importlib.util.spec_from_file_location("stats", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_rows(tmp_path, labels, probs, name="r", questions=("q",)):
    """Labels[i][qi] in {True, False}; probs[i][qi] = P(true). Rows via kev's own loaders and row builder."""
    from kev.benchmark import prediction_rows
    from kev.data import load_records
    f = tmp_path / f"{name}.jsonl"
    with f.open("w") as out:
        for i, lab in enumerate(labels):
            qs = {q: {"type": "noul", "instructions": "Synthetic?", "label": bool(lab[k])} for k, q in enumerate(questions)}
            out.write(json.dumps({"_meta": {"id": f"rec{i}", "group": f"g{i}"}, "state": "synthetic", "questions": qs}) + "\n")
    rows = []
    for rec, pr in zip(load_records(str(f)), probs):
        p = {q: {"true": pr[k], "false": 1 - pr[k]} for k, q in enumerate(questions)}
        z = {q: {"true": float(np.log(pr[k])), "false": float(np.log(1 - pr[k]))} for k, q in enumerate(questions)}
        rows += prediction_rows(rec, {"probabilities": p, "logits": z, "inference_temperature": 1.0})
    return rows


def test_macro_f1_perfect_wrong_and_absent_class(st, tmp_path):
    good = make_rows(tmp_path, [[True], [False], [True]], [[0.9], [0.2], [0.8]])
    assert st.macro_f1(good) == pytest.approx(1.0)
    bad = make_rows(tmp_path, [[True], [False]], [[0.1], [0.9]], name="bad")
    assert st.macro_f1(bad) == pytest.approx(0.0)
    only_true = make_rows(tmp_path, [[True], [True]], [[0.9], [0.1]], name="ot")   # class "false" absent from labels
    assert st.macro_f1(only_true) == pytest.approx(2 * 0.5 * 1.0 / 1.5)          # F1 of "true" only: P 1, R 0.5


def test_macro_f1_averages_over_questions(st, tmp_path):
    rows = make_rows(tmp_path, [[True, True], [False, True]], [[0.9, 0.9], [0.1, 0.9]], questions=("a", "b"))
    # question a: perfect -> 1.0; question b: only class "true" present, all predicted true -> 1.0
    assert st.macro_f1(rows) == pytest.approx(1.0)


def test_common_keeps_only_shared_examples_in_the_same_order(st, tmp_path):
    a = make_rows(tmp_path, [[True], [False], [True]], [[0.9], [0.2], [0.8]], name="a")
    b = make_rows(tmp_path, [[True], [False], [True]], [[0.7], [0.3], [0.6]], name="b")[1:]
    shared, dropped = st.common({"a": a, "b": b})
    assert [r["id"] for r in shared["a"]] == [r["id"] for r in shared["b"]] == ["rec1", "rec2"]
    assert dropped == {"a": 1, "b": 0}


def test_temperature_keeps_argmax_and_softens(st, tmp_path):
    rows = make_rows(tmp_path, [[True], [False]], [[0.9], [0.3]])
    hot = st.with_temperature(rows, 2.0)
    assert [np.argmax(r["p"]) for r in hot] == [np.argmax(r["p"]) for r in rows]
    assert max(hot[0]["p"]) < max(rows[0]["p"])
    assert st.with_temperature(rows, 1.0)[0]["p"] == pytest.approx(rows[0]["p"])


def test_bootstrap_is_seeded_and_zero_for_identical_runs(st, tmp_path):
    rng = random.Random(0)
    labels = [[rng.random() < 0.5] for _ in range(40)]
    probs = [[min(0.99, max(0.01, (0.8 if l[0] else 0.2) + rng.uniform(-0.3, 0.3)))] for l in labels]
    rows = make_rows(tmp_path, labels, probs)
    tasks = {"t1": {"kev": [rows], "sft": [rows]}}
    diff = lambda d: st.mean_over_tasks(d, lambda runs: st.seed_mean(runs["kev"], st.automation) - st.seed_mean(runs["sft"], st.automation))
    d1 = st.bootstrap(tasks, diff, samples=200, seed=5)
    assert np.allclose(d1, 0) and np.array_equal(d1, st.bootstrap(tasks, diff, samples=200, seed=5))


def test_bootstrap_resamples_whole_records(st, tmp_path):
    rows = make_rows(tmp_path, [[True, False]] * 10, [[0.9, 0.1]] * 10, questions=("a", "b"))
    seen = []
    def stat(d):
        seen.append(len(d["t"]["x"][0]))
        return 0.0
    st.bootstrap({"t": {"x": [rows]}}, stat, samples=5, seed=1)
    assert all(n % 2 == 0 for n in seen)   # both questions of a record always travel together


def test_h2_closure_is_a_ratio_of_mean_gaps(st):
    zs = {"t1": 0.5, "t2": 0.6}
    ft = {"t1": 0.8, "t2": 0.62}
    fr = {"t1": 0.9, "t2": 0.58}   # t2: zero shot already above the frontier (negative gap) stays defined
    assert st.h2_closure(zs, ft, fr) == pytest.approx(((0.3 + 0.02) / 2) / ((0.4 - 0.02) / 2))


def test_best_frontier_mode_per_task(st):
    modes = {"nothink-k0": {"p1": 0.80, "p2": 0.70}, "think-k3": {"p1": 0.85, "p2": 0.65}}
    assert st.best_mode(modes) == {"p1": 0.85, "p2": 0.70}


def test_holm_adjustment(st):
    adj = st.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_ci_and_one_sided_p(st):
    draws = np.array([0.1, 0.2, -0.05, 0.3, 0.15])
    lo, hi = st.ci(draws)
    assert lo < hi and st.p_greater_than_zero(draws) == pytest.approx(1 / 5)


def random_rows(tmp_path, n, seed, name, questions=("a", "b")):
    rng = random.Random(seed)
    labels = [[rng.random() < 0.6 for _ in questions] for _ in range(n)]
    probs = [[round(min(0.99, max(0.01, (0.75 if l else 0.3) + rng.uniform(-0.4, 0.4))), 2) for l in lab] for lab in labels]
    return make_rows(tmp_path, labels, probs, name=name, questions=questions)   # rounding creates confidence ties


def test_array_metrics_equal_kev_metrics_including_ties(st, tmp_path):
    from kev.metrics import metrics
    rows = random_rows(tmp_path, 300, 3, "eq")
    a = st.arrays(rows)
    m = metrics(rows)
    assert st.automation(a) == pytest.approx(m["coverage_at_5pct_error"])
    assert st.accuracy(a) == pytest.approx(m["acc"])
    assert st.macro_f1(a) == pytest.approx(st.macro_f1(rows))


def test_take_and_arrays_round_trip(st, tmp_path):
    rows = random_rows(tmp_path, 20, 4, "rt")
    a = st.arrays(rows)
    idx = np.array([0, 0, 5, 7])
    t = st.take(a, idx)
    assert t.conf.tolist() == a.conf[idx].tolist() and len(t) == 4


def test_bootstrap_draw_is_the_point_estimate_with_identity_resampling(st, tmp_path):
    rows = random_rows(tmp_path, 50, 9, "pe")
    tasks = {"t": {"x": [rows]}}
    point = st.automation(st.arrays(rows))
    draws = st.bootstrap(tasks, lambda d: st.automation(d["t"]["x"][0]), samples=300, seed=2)
    assert min(draws) <= point <= max(draws)


def test_array_ece_and_brier_equal_kev_metrics(st, tmp_path):
    from kev.metrics import metrics
    rows = random_rows(tmp_path, 300, 11, "eb")
    m = metrics(rows)
    a = st.arrays(rows)
    assert st.ece(a) == pytest.approx(m["ece"]) and st.brier(a) == pytest.approx(m["brier"])
    t = st.take(a, np.arange(10))
    assert len(t.brier) == 10
