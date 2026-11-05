# /// script
# requires-python = ">=3.11"
# ///
"""Post hoc analysis: per-task paired differences
against the frontier with 95% CIs.

For each task and each frontier model, the frontier row is the model's best-of-four mode by test macro-F1 (the same
choice H2 makes, scripts/analysis/hypotheses.py). Differences Kev-4B 3k minus frontier and LoRA-SFT 3k minus frontier (seed
means over the three seeds) in accuracy and macro-F1, on the pairs all runs scored, with a record-cluster bootstrap
(protocol seed, 10,000 samples; all questions of a record together, one draw shared by every run and seed). Reported:
point, 95% percentile CI, two-sided bootstrap P. Output: runs/pertask-frontier.json (aggregates only)."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import hypotheses as hy  # noqa: E402
import stats as st  # noqa: E402

METRICS = {"accuracy": st.accuracy, "macro_f1": st.macro_f1}
STATS = [(row, m) for row in ("kev", "sft") for m in METRICS]


def record_index(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Start and length of each record's contiguous block of rows (aligned rows are sorted by (id, question))."""
    ids = [r["id"] for r in rows]
    starts = [i for i in range(len(ids)) if i == 0 or ids[i] != ids[i - 1]]
    if len(starts) != len(set(ids)):
        raise ValueError("rows of a record are not contiguous")
    s = np.array(starts)
    return s, np.diff(np.append(s, len(ids)))


def draw_index(starts: np.ndarray, lens: np.ndarray, picks: np.ndarray) -> np.ndarray:
    """Row indices of the picked records, in pick order (same as concatenating each picked record's span)."""
    ln = lens[picks]
    offs = np.repeat(np.cumsum(ln) - ln, ln)
    return np.repeat(starts[picks], ln) + (np.arange(ln.sum()) - offs)


def fast_bootstrap(tasks, stat, samples: int = st.BOOTSTRAP_SAMPLES, seed: int = st.BOOTSTRAP_SEED) -> np.ndarray:
    """St.bootstrap with the same draws and the same resampled rows, using contiguous record blocks instead of
    concatenating one span per pick (identical output; tests/test_g4_optional.py checks it against st.bootstrap)."""
    rng = np.random.default_rng(seed)
    index, arr = {}, {}
    for t, runs in tasks.items():
        ref = next(iter(runs.values()))[0]
        for seeds in runs.values():
            for rows in seeds:
                if [st.key(r) for r in rows] != [st.key(r) for r in ref]:
                    raise ValueError(f"{t}: runs are not aligned; call common() first")
        index[t] = record_index(ref)
        arr[t] = {name: [st.arrays(rows) for rows in seeds] for name, seeds in runs.items()}
    out = np.empty(samples)
    for b in range(samples):
        draw = {}
        for t in tasks:
            starts, lens = index[t]
            idx = draw_index(starts, lens, rng.integers(0, len(starts), len(starts)))
            draw[t] = {name: [st.take(a, idx) for a in seeds] for name, seeds in arr[t].items()}
        out[b] = stat(draw)
    return out


def two_sided(draws: np.ndarray) -> float:
    pk = float(np.mean(draws <= 0))
    return min(1.0, 2 * min(pk, 1 - pk))


def differences(arr: dict) -> np.ndarray:
    fr = {m: fn(arr["fr"][0]) for m, fn in METRICS.items()}
    return np.array([st.seed_mean(arr[row], METRICS[m]) - fr[m] for row, m in STATS])


def job(args):
    task, model, mode, samples, seed = args
    frozen = json.loads((ROOT / "configs" / "frozen.json").read_text())
    named = {"kev": hy.gather(ROOT, frozen, "kev4b-3k", task, "test"), "sft": hy.gather(ROOT, frozen, "sft4b-3k", task, "test"),
             "fr": [hy.frontier_rows(ROOT, model, mode, task)]}
    aligned, dropped = hy.align(named)
    arr = {k: [st.arrays(r) for r in v] for k, v in aligned.items()}
    point = differences(arr)
    starts, lens = record_index(aligned["kev"][0])
    rng = np.random.default_rng(seed)
    draws = np.empty((samples, len(STATS)))
    for b in range(samples):
        idx = draw_index(starts, lens, rng.integers(0, len(starts), len(starts)))
        draws[b] = differences({k: [st.take(a, idx) for a in v] for k, v in arr.items()})
    levels = {f"{row}_{m}": float(st.seed_mean(arr[row], METRICS[m])) for row in ("kev", "sft") for m in METRICS}
    levels.update({f"fr_{m}": float(fn(arr["fr"][0])) for m, fn in METRICS.items()})
    out = {"mode": mode, "n_pairs": len(aligned["kev"][0]), "n_records": len(starts), "dropped_pairs": dropped, "levels": levels}
    for i, (row, m) in enumerate(STATS):
        lo, hi = st.ci(draws[:, i])
        out[f"{row}_minus_fr_{m}"] = {"point": float(point[i]), "ci95": [lo, hi], "p_two_sided": two_sided(draws[:, i])}
    print(task, model, mode, {k: round(v["point"], 4) for k, v in out.items() if isinstance(v, dict) and "point" in v}, flush=True)
    return task, hy.FRONTIER[model], out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--samples", type=int, default=st.BOOTSTRAP_SAMPLES)
    ap.add_argument("--seed", type=int, default=st.BOOTSTRAP_SEED)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", default="runs/pertask-frontier.json")
    a = ap.parse_args(argv)
    h2 = json.loads((ROOT / "runs" / "hypotheses-test.json").read_text())["H2"]["models"]
    long = {v: k for k, v in hy.FRONTIER.items()}
    plan = [(t, long[short], h2[short]["per_task"][t]["frontier_mode"], a.samples, a.seed) for short in h2 for t in hy.TASKS]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        res = list(ex.map(job, plan))
    out = {"note": __doc__.split("\n\n")[0], "samples": a.samples, "seed": a.seed, "tasks": {}}
    for t, short, r in res:
        out["tasks"].setdefault(t, {})[short] = r
    (ROOT / a.out).write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
