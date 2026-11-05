"""Scripts/analysis/hypotheses.py on a synthetic repo: frozen entries pointing at synthetic rows built with kev's own row
builder. Checks the verdict logic, incompleteness handling and the pairing rules, not real results."""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

ANALYSIS = Path(__file__).resolve().parents[1] / "scripts" / "analysis"
sys.path.insert(0, str(ANALYSIS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ANALYSIS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # worker processes look functions up by module name
    spec.loader.exec_module(mod)
    return mod


hy = load("hypotheses")


def rows_for(tmp, quality, n=200, seed=0, drop=0):
    """Noul rows. `quality` = (accuracy, confidence when right, confidence when wrong): a model that is right more often
    and unsure when wrong automates more at a 5% error budget."""
    import math
    from kev.benchmark import prediction_rows
    from kev.data import load_records
    acc, c_right, c_wrong = quality
    rng = random.Random(seed * 1000 + int(acc * 100))
    f = tmp / f"recs{seed}-{acc}-{c_right}-{c_wrong}-{drop}.jsonl"
    labels = [random.Random(i).random() < 0.5 for i in range(n)]
    with f.open("w") as out:
        for i, lab in enumerate(labels[drop:], start=drop):
            out.write(json.dumps({"_meta": {"id": f"r{i}", "group": f"g{i}"}, "state": "synthetic",
                                  "questions": {"q": {"type": "noul", "instructions": "x", "label": lab}}}) + "\n")
    rows = []
    for rec in load_records(str(f)):
        lab = rec["questions"]["q"]["label"]
        right = rng.random() < acc
        c = min(0.99, max(0.51, (c_right if right else c_wrong) + rng.uniform(-0.03, 0.03)))
        p_true = c if (lab == right) else 1 - c
        rows += prediction_rows(rec, {"probabilities": {"q": {"true": p_true, "false": 1 - p_true}},
                                      "logits": {"q": {"true": math.log(p_true), "false": math.log(1 - p_true)}},
                                      "inference_temperature": 1.0})
    return rows


GOOD, POOR, WEAK, STRONG = (0.95, 0.97, 0.6), (0.88, 0.92, 0.92), (0.7, 0.8, 0.8), (0.92, 0.95, 0.7)


def write_run(root, run, split, rows):
    d = root / f"{run}-{split}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "rows.json").write_text(json.dumps(rows))


def repo(tmp_path, kev_q=GOOD, sft_q=POOR, tasks=hy.TASKS, sft_tasks=None, drop_sft=0):
    root = tmp_path
    entries = {}
    for t in tasks:
        for s in hy.SEEDS:
            for row, q, run in (("kev4b-3k", kev_q, f"runs/kev4b-{t}-3k-s{s}"), ("sft4b-3k", sft_q, f"runs/sft4b-{t}-3k-s{s}")):
                if row == "sft4b-3k" and sft_tasks is not None and t not in sft_tasks:
                    continue
                write_run(root, run, "test", rows_for(tmp_path, q, seed=s, drop=drop_sft if row == "sft4b-3k" else 0))
                entries[f"{row}/{t}/s{s}"] = {"run": run, "dev_temperature": 1.0}
        for row, q, run in (("kev4b-zeroshot", WEAK, f"runs/kev4b-{t}-zeroshot"), ("qwen4b-zeroshot-k0", (0.75, 0.8, 0.8), f"runs/qwen4b-{t}-zeroshot-k0")):
            write_run(root, run, "test", rows_for(tmp_path, q))
            entries[f"{row}/{t}/s0"] = {"run": run, "dev_temperature": 1.0}
    (root / "configs").mkdir(exist_ok=True)
    (root / "configs" / "frozen.json").write_text(json.dumps({"entries": entries}))
    return root, json.loads((root / "configs" / "frozen.json").read_text())


def test_h3_passes_when_kev_automates_clearly_more(tmp_path):
    root, frozen = repo(tmp_path)
    r = hy.h3(root, frozen, "test", hy.TASKS, samples=200, seed=1)
    assert r["status"] == "ok" and r["primary_mean_difference"]["point"] > 0 and r["passes"]
    assert set(r["secondary"]) == {"accuracy", "macro_f1", "ece", "brier"} and "p_holm" in r["secondary"]["accuracy"]


def test_h3_fails_when_sft_is_better(tmp_path):
    root, frozen = repo(tmp_path, kev_q=POOR, sft_q=GOOD)
    r = hy.h3(root, frozen, "test", hy.TASKS, samples=200, seed=1)
    assert r["status"] == "ok" and not r["passes"] and r["primary_mean_difference"]["point"] < 0


def test_h3_is_incomplete_without_every_task(tmp_path):
    root, frozen = repo(tmp_path, sft_tasks=("p1", "p2"))
    assert hy.h3(root, frozen, "test", hy.TASKS, samples=50, seed=1)["status"] == "incomplete"


def test_h3_pairs_on_common_records_and_reports_the_drop(tmp_path):
    root, frozen = repo(tmp_path, drop_sft=4)
    r = hy.h3(root, frozen, "test", ("p1",), samples=50, seed=1)
    assert r["dropped_pairs"]["p1"] == 4


def test_h1_ratio_of_mean_macro_f1(tmp_path):
    root, frozen = repo(tmp_path)
    r = hy.h1(root, frozen, "test", samples=100, seed=1)
    assert r["status"] == "ok" and r["threshold"] == 0.90
    lo, hi = r["ratio_kev_over_qwen"]["ci95"]
    assert lo <= r["ratio_kev_over_qwen"]["point"] <= hi


def test_h2_is_incomplete_without_all_frontier_modes(tmp_path):
    root, frozen = repo(tmp_path)
    r = hy.h2(root, frozen, samples=50, seed=1)
    assert r["status"] == "incomplete" and r["models"]["deepseek"]["status"] == "incomplete"


def test_h2_closure_with_all_modes(tmp_path):
    root, frozen = repo(tmp_path)
    for model in hy.FRONTIER:
        for m in hy.FRONTIER_MODES:
            for t in hy.TASKS:
                q = STRONG if m == "nothink-k3" else WEAK
                write_run(root, f"runs/frontier-{model}-{m}-{t}", "test", rows_for(tmp_path, q, seed=7))
    r = hy.h2(root, frozen, samples=100, seed=1)
    assert r["status"] == "ok"
    ds = r["models"]["deepseek"]
    assert all(ds["per_task"][t]["frontier_mode"] == "nothink-k3" for t in hy.TASKS)
    assert ds["passes"] == (ds["closure"]["point"] >= 0.5)


def test_parallel_series_give_exactly_the_serial_draws(tmp_path):
    root, frozen = repo(tmp_path, tasks=("p1", "p2"))
    serial = hy.h3(root, frozen, "test", ("p1", "p2"), samples=60, seed=3, jobs=1)
    parallel = hy.h3(root, frozen, "test", ("p1", "p2"), samples=60, seed=3, jobs=4)
    assert serial == parallel


def test_only_recomputes_the_named_hypotheses_and_keeps_the_rest(tmp_path, monkeypatch):
    root, frozen = repo(tmp_path, tasks=("p1",))
    out = tmp_path / "h.json"
    out.write_text(json.dumps({"split": "test", "H1": {"status": "kept"}, "H3": {"status": "kept"}}))
    monkeypatch.setattr(hy, "h2", lambda *a, **k: {"status": "recomputed"})
    hy.main(["--root", str(root), "--only", "H2", "--out", str(out), "--samples", "10"])
    r = json.loads(out.read_text())
    assert r["H2"] == {"status": "recomputed"} and r["H1"] == {"status": "kept"} and r["H3"] == {"status": "kept"}


def test_h3_secondary_has_all_four_preregistered_metrics_with_the_right_direction(tmp_path):
    root, frozen = repo(tmp_path)                       # Kev better and better calibrated than SFT
    r = hy.h3(root, frozen, "test", hy.TASKS, samples=200, seed=1)
    assert set(r["secondary"]) == {"accuracy", "macro_f1", "ece", "brier"}
    # Brier: Kev (right more often, rarely confident when wrong) is clearly better; lower is better, so Kev - SFT < 0
    assert r["secondary"]["brier"]["point"] < 0 and r["secondary"]["brier"]["p_kev_better"] < 0.05
    assert r["secondary"]["ece"]["lower_is_better"] and r["secondary"]["brier"]["lower_is_better"]
    assert not r["secondary"]["accuracy"]["lower_is_better"]
    # direction: p_kev_better for a lower-is-better metric is the share of draws with Kev - SFT >= 0
    e = r["secondary"]["ece"]
    assert (e["point"] < 0) == (e["p_kev_better"] < 0.5)
    assert all("p_holm" in v for v in r["secondary"].values())
