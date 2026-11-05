# /// script
# requires-python = ">=3.11"
# ///
"""Compute H1, H2, H3 and H4b exactly as fixed in the statistical protocol of 2026-10-18.

Reads configs/frozen.json and each frozen run's scored rows (`<run>-test/rows.json`, or `<run>-dev/rows.json` with
--split dev for a dry run), applies each run's dev-fitted temperature, restricts every comparison to the pairs all its
runs scored, and bootstraps over records (scripts/analysis/stats.py). Frontier rows (test only) come from
runs/frontier-<model>-<mode>-<k>-<task>-test/rows.json, used raw. Writes one JSON with every point estimate, CI,
p-value, dropped-pair count and verdict; prints a short summary. Reads rows.json only, never row content."""
from __future__ import annotations

import argparse
import functools
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats as st  # noqa: E402

TASKS = ("p1", "p2", "p3", "i1", "i2", "i3")
INTERNAL = ("i1", "i2", "i3")
SEEDS = (0, 1, 2)
FRONTIER = {"deepseek-v4.1-flash": "deepseek", "kimi-k3": "kimi"}
FRONTIER_MODES = ("nothink-k0", "nothink-k3", "think-k0", "think-k3")
SECONDARY = {"accuracy": st.accuracy, "macro_f1": st.macro_f1, "ece": st.ece, "brier": st.brier}   # protocol item 7
LOWER_IS_BETTER = {"ece", "brier"}
METRICS = {"automation_at_5pct": st.automation, **SECONDARY}
_SHARED: dict = {}   # aligned rows per task, set before forking the bootstrap workers (fork shares, nothing is pickled)


def _kev_minus_sft(d, metric):
    fn = METRICS[metric]
    return st.mean_over_tasks(d, lambda r: st.seed_mean(r["kev"], fn) - st.seed_mean(r["sft"], fn))


def _series(job):
    tasks, metric, samples, seed = job
    return st.bootstrap({t: _SHARED[t] for t in tasks}, functools.partial(_kev_minus_sft, metric=metric), samples, seed)


def run_series(jobs: list, n_jobs: int) -> list:
    """Independent bootstrap series (each with the protocol seed), serially or in forked worker processes; identical draws."""
    if n_jobs <= 1:
        return [_series(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=min(n_jobs, len(jobs)), mp_context=mp.get_context("fork")) as ex:
        return list(ex.map(_series, jobs))


def run_rows(root: Path, entry: dict, split: str) -> list[dict]:
    rows = st.load_rows(root / f"{entry['run']}-{split}")
    return st.with_temperature(rows, entry["dev_temperature"])


def frontier_rows(root: Path, model: str, mode: str, task: str) -> list[dict] | None:
    d = root / "runs" / f"frontier-{model}-{mode}-{task}-test"
    return st.load_rows(d) if (d / "rows.json").exists() else None


def gather(root: Path, frozen: dict, row: str, task: str, split: str, seeds=SEEDS) -> list[list[dict]] | None:
    out = []
    for s in seeds:
        e = frozen["entries"].get(f"{row}/{task}/s{s}")
        if e is None:
            return None
        out.append(run_rows(root, e, split))
    return out


def align(named: dict[str, list[list[dict]]]) -> tuple[dict[str, list[list[dict]]], int]:
    """Restrict every seed of every run to the pairs all of them scored; returns the aligned runs and the largest drop."""
    flat = {f"{name}#{i}": rows for name, seeds in named.items() for i, rows in enumerate(seeds)}
    shared, dropped = st.common(flat)
    out = {name: [shared[f"{name}#{i}"] for i in range(len(seeds))] for name, seeds in named.items()}
    return out, max(dropped.values())


def summarize(draws: np.ndarray, point: float) -> dict:
    lo, hi = st.ci(draws)
    return {"point": point, "ci95": [lo, hi], "p_one_sided_gt0": st.p_greater_than_zero(draws)}


def h3(root, frozen, split, tasks, samples, seed, jobs=1) -> dict:
    per, dropped = {}, {}
    for t in tasks:
        kev, sft = gather(root, frozen, "kev4b-3k", t, split), gather(root, frozen, "sft4b-3k", t, split)
        if kev is None or sft is None:
            return {"status": "incomplete", "missing_task": t}
        aligned, dropped[t] = align({"kev": kev, "sft": sft})
        _SHARED[t] = aligned
        per[t] = {m: {"kev": st.seed_mean(aligned["kev"], fn), "sft": st.seed_mean(aligned["sft"], fn)} for m, fn in METRICS.items()}
    tasks = tuple(tasks)
    plan = ([(tasks, "automation_at_5pct", samples, seed)] + [(tasks, m, samples, seed) for m in SECONDARY]
            + [((t,), "automation_at_5pct", samples, seed) for t in tasks])
    draws = run_series(plan, jobs)
    point = lambda m, ts: float(sum(per[t][m]["kev"] - per[t][m]["sft"] for t in ts) / len(ts))
    primary = summarize(draws[0], point("automation_at_5pct", tasks))
    secondary, p = {}, {}
    for i, m in enumerate(SECONDARY, start=1):
        secondary[m] = summarize(draws[i], point(m, tasks))   # point and CI: Kev minus SFT
        # one-sided p for "Kev is better": the share of resamples where Kev is not better
        p[m] = float(np.mean(draws[i] >= 0)) if m in LOWER_IS_BETTER else st.p_greater_than_zero(draws[i])
        secondary[m]["p_kev_better"] = p[m]
        secondary[m]["lower_is_better"] = m in LOWER_IS_BETTER
    for m, adj in st.holm(p).items():
        secondary[m]["p_holm"] = adj
    k = 1 + len(SECONDARY)
    per_task_ci = {t: summarize(draws[k + i], point("automation_at_5pct", (t,))) for i, t in enumerate(tasks)}
    return {"status": "ok", "tasks": list(tasks), "per_task": per, "per_task_primary": per_task_ci,
            "primary_mean_difference": primary, "secondary": secondary, "dropped_pairs": dropped,
            "passes": primary["ci95"][0] > 0}


def h1(root, frozen, split, samples, seed) -> dict:
    per, aligned_all, dropped = {}, {}, {}
    for t in TASKS:
        kev, qwen = gather(root, frozen, "kev4b-zeroshot", t, split, (0,)), gather(root, frozen, "qwen4b-zeroshot-k0", t, split, (0,))
        if kev is None or qwen is None:
            return {"status": "incomplete", "missing_task": t}
        aligned, dropped[t] = align({"kev": kev, "qwen": qwen})
        aligned_all[t] = aligned
        per[t] = {"kev": st.macro_f1(aligned["kev"][0]), "qwen": st.macro_f1(aligned["qwen"][0])}
    ratio = lambda d: (st.mean_over_tasks(d, lambda r: st.macro_f1(r["kev"][0]))
                       / st.mean_over_tasks(d, lambda r: st.macro_f1(r["qwen"][0])))
    point = float(np.mean([per[t]["kev"] for t in TASKS]) / np.mean([per[t]["qwen"] for t in TASKS]))
    draws = st.bootstrap(aligned_all, ratio, samples, seed)
    lo, hi = st.ci(draws)
    return {"status": "ok", "per_task_macro_f1": per, "ratio_kev_over_qwen": {"point": point, "ci95": [lo, hi]},
            "threshold": 0.90, "accuracy_part_passes": point >= 0.90, "dropped_pairs": dropped,
            "note": "descriptive with 95% CIs; the latency part is reported separately (pinned measurement)"}


def h2(root, frozen, samples, seed) -> dict:
    """Test split only (the frontier rows exist on test only)."""
    zs, ft, out = {}, {}, {"status": "ok", "models": {}}
    kev = {t: (gather(root, frozen, "kev4b-zeroshot", t, "test", (0,)), gather(root, frozen, "kev4b-3k", t, "test")) for t in TASKS}
    if any(a is None or b is None for a, b in kev.values()):
        return {"status": "incomplete", "missing": "kev rows"}
    for model, short in FRONTIER.items():
        modes = {m: {t: frontier_rows(root, model, m, t) for t in TASKS} for m in FRONTIER_MODES}
        missing = [f"{m}/{t}" for m in modes for t in TASKS if modes[m][t] is None]
        if missing:
            out["models"][short] = {"status": "incomplete", "missing_frontier_rows": missing}
            out["status"] = "incomplete"
            continue
        best_mode, aligned_all, dropped, per = {}, {}, {}, {}
        for t in TASKS:
            named = {"zs": kev[t][0], "ft": kev[t][1], **{m: [modes[m][t]] for m in FRONTIER_MODES}}
            aligned, dropped[t] = align(named)
            f1 = {m: st.macro_f1(aligned[m][0]) for m in FRONTIER_MODES}
            best_mode[t] = max(f1, key=f1.get)          # selected once on the full test set, kept fixed in resamples
            aligned_all[t] = {"zs": aligned["zs"], "ft": aligned["ft"], "fr": aligned[best_mode[t]]}
            per[t] = {"zero_shot": st.macro_f1(aligned["zs"][0]), "fine_tuned": st.seed_mean(aligned["ft"], st.macro_f1),
                      "frontier": f1[best_mode[t]], "frontier_mode": best_mode[t], "frontier_all_modes": f1}
        closure = lambda d: st.h2_closure({t: st.macro_f1(d[t]["zs"][0]) for t in d},
                                          {t: st.seed_mean(d[t]["ft"], st.macro_f1) for t in d},
                                          {t: st.macro_f1(d[t]["fr"][0]) for t in d})
        point = st.h2_closure({t: per[t]["zero_shot"] for t in TASKS}, {t: per[t]["fine_tuned"] for t in TASKS},
                              {t: per[t]["frontier"] for t in TASKS})
        draws = st.bootstrap(aligned_all, closure, samples, seed)
        lo, hi = st.ci(draws)
        out["models"][short] = {"status": "ok", "per_task": per, "closure": {"point": point, "ci95": [lo, hi]},
                                "threshold": 0.5, "passes": point >= 0.5, "dropped_pairs": dropped}
    if out["status"] == "ok":
        out["passes"] = all(v["passes"] for v in out["models"].values())
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", choices=("test", "dev"), default="test")
    ap.add_argument("--root", default=".")
    ap.add_argument("--samples", type=int, default=st.BOOTSTRAP_SAMPLES)
    ap.add_argument("--seed", type=int, default=st.BOOTSTRAP_SEED)
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=1, help="parallel bootstrap series (forked processes; identical draws)")
    ap.add_argument("--only", help="comma list of H1,H2,H3,H4b to (re)compute; the others are kept from --out")
    a = ap.parse_args(argv)
    root = Path(a.root)
    frozen = json.loads((root / "configs" / "frozen.json").read_text())
    out = Path(a.out or f"runs/hypotheses-{a.split}.json")
    compute = {"H1": lambda: h1(root, frozen, a.split, a.samples, a.seed),
               "H3": lambda: h3(root, frozen, a.split, TASKS, a.samples, a.seed, a.jobs),
               "H4b": lambda: h3(root, frozen, a.split, INTERNAL, a.samples, a.seed, a.jobs),
               "H2": lambda: h2(root, frozen, a.samples, a.seed) if a.split == "test" else {"status": "test only"}}
    names = a.only.split(",") if a.only else list(compute)
    result = json.loads(out.read_text()) if (a.only and out.exists()) else {}
    result.update({"split": a.split, "samples": a.samples, "seed": a.seed})
    for h in names:
        result[h] = compute[h]()
    out.write_text(json.dumps(result, indent=1) + "\n")
    for h in names:
        r = result[h]
        print(h, r.get("status"), {k: r[k] for k in ("passes", "accuracy_part_passes") if k in r})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
