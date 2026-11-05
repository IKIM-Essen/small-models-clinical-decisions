# /// script
# requires-python = ">=3.11"
# ///
"""Statistics.

Building blocks used by the hypothesis computations: kev's own row metrics (kev.metrics.metrics, raw_row, tempered_row)
so every number matches kev.benchmark; macro-F1 per question averaged over a task's questions; paired comparisons on the
(record, question) pairs every compared run scored; seed means; a cluster bootstrap over records (all questions of a
record together) resampled jointly for every run and seed of a task; 95% percentile CIs; Holm adjustment.
Reads rows.json / calibration.json only; never prints row content."""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from kev.metrics import coverage_at_error, ece as kev_ece, metrics, raw_row, scored_rows, tempered_row

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20261018


def load_rows(run_dir: str | Path) -> list[dict]:
    return scored_rows(json.loads((Path(run_dir) / "rows.json").read_text()))


def dev_temperature(dev_dir: str | Path) -> float:
    """The temperature kev.calibrate fitted on the run's own dev rows (protocol item 1)."""
    return float(json.loads((Path(dev_dir) / "calibration.json").read_text())["workload_temperature"])


def with_temperature(rows: list[dict], temperature: float) -> list[dict]:
    if not rows or "logits" not in rows[0]:
        return rows                               # remote rows without logits are used raw
    return [tempered_row(raw_row(r), temperature) for r in rows]


def key(row: dict) -> tuple[str, str]:
    return row["id"], row["question"]


def common(row_sets: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], dict[str, int]]:
    """Each run restricted to the (record, question) pairs all runs scored, in one shared order."""
    shared = set.intersection(*(set(map(key, rows)) for rows in row_sets.values()))
    order = sorted(shared)
    out, dropped = {}, {}
    for name, rows in row_sets.items():
        index = {key(r): r for r in rows}
        out[name] = [index[k] for k in order]
        dropped[name] = len(rows) - len(order)
    return out, dropped


@dataclass
class Arrays:
    """One run's scored rows as aligned vectors: top-probability confidence, correctness, prediction, label, question."""
    conf: np.ndarray
    correct: np.ndarray
    pred: np.ndarray
    label: np.ndarray
    question: np.ndarray
    brier: np.ndarray        # per-row Brier score, sum over options of (p - one-hot)^2 (kev.metrics' definition)

    def __len__(self) -> int:
        return len(self.conf)


def arrays(rows: list[dict]) -> Arrays:
    p = [np.asarray(r["p"], dtype=float) for r in rows]
    pred = np.array([int(x.argmax()) for x in p])
    label = np.array([int(r["label"]) for r in rows])
    brier = np.array([float(((x - np.eye(len(x))[y]) ** 2).sum()) for x, y in zip(p, label)])
    return Arrays(conf=np.array([float(x.max()) for x in p]), correct=pred == label, pred=pred, label=label,
                  question=np.array([r["question"] for r in rows]), brier=brier)


def take(a: Arrays, idx: np.ndarray) -> Arrays:
    return Arrays(a.conf[idx], a.correct[idx], a.pred[idx], a.label[idx], a.question[idx], a.brier[idx])


def _as_arrays(x) -> Arrays:
    return x if isinstance(x, Arrays) else arrays(x)


def macro_f1(x) -> float:
    a = _as_arrays(x)
    scores = []
    for q in np.unique(a.question):
        m = a.question == q
        y, yhat = a.label[m], a.pred[m]
        f1s = []
        for c in np.unique(y):
            tp = int(((yhat == c) & (y == c)).sum())
            fp = int(((yhat == c) & (y != c)).sum())
            fn = int(((yhat != c) & (y == c)).sum())
            f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
        scores.append(float(np.mean(f1s)))
    return float(np.mean(scores))


def accuracy(x) -> float:
    return float(_as_arrays(x).correct.mean())


def ece(x) -> float:
    a = _as_arrays(x)
    return float(kev_ece(a.conf.tolist(), a.correct.astype(int).tolist()))


def brier(x) -> float:
    return float(_as_arrays(x).brier.mean())


def automation(x, budget: float = 0.05) -> float:
    """Kev's own coverage_at_error (descending confidence, ties as kev handles them) on the run's vectors."""
    a = _as_arrays(x)
    return float(coverage_at_error(a.conf, a.correct, budget))


def seed_mean(seed_runs: list, fn) -> float:
    return float(np.mean([fn(r) for r in seed_runs]))


def mean_over_tasks(tasks: dict, fn) -> float:
    return float(np.mean([fn(runs) for runs in tasks.values()]))


def _record_spans(rows: list[dict]) -> list[np.ndarray]:
    spans: "OrderedDict[str, list[int]]" = OrderedDict()
    for i, r in enumerate(rows):
        spans.setdefault(r["id"], []).append(i)
    return [np.array(v) for v in spans.values()]


def bootstrap(tasks: dict[str, dict[str, list[list[dict]]]], stat, samples: int = BOOTSTRAP_SAMPLES,
              seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """Tasks[task][run] = [rows of seed 0, seed 1, ...], all aligned (same pairs, same order; use `common` first).
    Each resample draws record ids with replacement per task and applies the same draw to every run and seed; `stat`
    receives the same structure with each seed's rows as `Arrays`."""
    rng = np.random.default_rng(seed)
    spans, arr = {}, {}
    for t, runs in tasks.items():
        ref = next(iter(runs.values()))[0]
        for seeds in runs.values():
            for rows in seeds:
                if [key(r) for r in rows] != [key(r) for r in ref]:
                    raise ValueError(f"{t}: runs are not aligned; call common() first")
        spans[t] = _record_spans(ref)
        arr[t] = {name: [arrays(rows) for rows in seeds] for name, seeds in runs.items()}
    out = np.empty(samples)
    for b in range(samples):
        draw = {}
        for t in tasks:
            picks = rng.integers(0, len(spans[t]), len(spans[t]))
            idx = np.concatenate([spans[t][p] for p in picks])
            draw[t] = {name: [take(a, idx) for a in seeds] for name, seeds in arr[t].items()}
        out[b] = stat(draw)
    return out


def ci(draws: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(np.quantile(draws, a)), float(np.quantile(draws, 1 - a))


def p_greater_than_zero(draws: np.ndarray) -> float:
    """One-sided bootstrap p-value for 'difference > 0': share of resamples at or below zero."""
    return float(np.mean(draws <= 0))


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    names = sorted(pvalues, key=pvalues.get)
    m, running, out = len(names), 0.0, {}
    for i, n in enumerate(names):
        running = max(running, min(1.0, (m - i) * pvalues[n]))
        out[n] = running
    return out


def best_mode(modes: dict[str, dict[str, float]]) -> dict[str, float]:
    """Per task, the best value over a frontier model's modes (H2: the strictest reference for Kev)."""
    tasks = set.intersection(*(set(v) for v in modes.values()))
    return {t: max(v[t] for v in modes.values()) for t in sorted(tasks)}


def h2_closure(zero_shot: dict[str, float], fine_tuned: dict[str, float], frontier: dict[str, float]) -> float:
    """Ratio of mean gaps over tasks: mean(ft - zs) / mean(frontier - zs) (protocol item 6)."""
    tasks = sorted(frontier)
    num = np.mean([fine_tuned[t] - zero_shot[t] for t in tasks])
    den = np.mean([frontier[t] - zero_shot[t] for t in tasks])
    return float(num / den)
