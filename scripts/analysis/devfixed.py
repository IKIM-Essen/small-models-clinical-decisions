# /// script
# requires-python = ">=3.11"
# ///
"""Sensitivity analysis: automation with a threshold fixed on development rows.

The primary H3/H4b endpoint chooses the confidence threshold on the evaluated (test) rows themselves, which is kev's
coverage_at_error rule and in-sample. Here, for every frozen fine-tuned row (Kev-4B 3k, LoRA-SFT 3k; 3 seeds), the
threshold is the lowest top-class confidence at which the run's own dev rows (with the run's dev temperature) stay at or
below the 5% error budget; it is then applied unchanged to the test rows, and test coverage and realized test error are
reported. H3 and H4b are recomputed under this rule as a declared sensitivity analysis, with the same paired record
bootstrap as the primary analysis (thresholds stay fixed inside the bootstrap). Output: runs/devfixed-test.json.
Declared after the primary results existed; it never replaces them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import hypotheses as hy  # noqa: E402
import stats as st  # noqa: E402

BUDGET = 0.05
ROWS = ("kev4b-3k", "sft4b-3k")


def dev_threshold(dev_rows: list[dict], budget: float = BUDGET) -> float:
    """Lowest confidence at which the dev rows, taken in descending confidence, stay at or below `budget` error.
    Mirrors kev's coverage_at_error sweep; returns +inf when no prefix qualifies (coverage 0)."""
    a = st.arrays(dev_rows)
    order = np.argsort(-a.conf, kind="stable")
    conf, correct = a.conf[order], a.correct[order]
    errors = np.cumsum(~correct)
    n = np.arange(1, len(conf) + 1)
    ok = errors / n <= budget
    # ties: a prefix may only end where the confidence changes, as in kev's rule
    boundary = np.r_[conf[1:] != conf[:-1], True]
    good = np.where(ok & boundary)[0]
    return float(conf[good[-1]]) if len(good) else float("inf")


def fixed_coverage(a: st.Arrays, thr: float) -> float:
    return float(np.mean(a.conf >= thr)) if np.isfinite(thr) else 0.0


def fixed_error(a: st.Arrays, thr: float) -> float | None:
    sel = a.conf >= thr
    return float(np.mean(~a.correct[sel])) if sel.any() else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--samples", type=int, default=st.BOOTSTRAP_SAMPLES)
    ap.add_argument("--seed", type=int, default=st.BOOTSTRAP_SEED)
    ap.add_argument("--out", default="runs/devfixed-test.json")
    a = ap.parse_args(argv)
    frozen = json.loads((ROOT / "configs" / "frozen.json").read_text())
    thresholds, per_task, aligned_all = {}, {}, {}
    for t in hy.TASKS:
        named, thr = {}, {}
        for row in ROWS:
            seeds = []
            for s in hy.SEEDS:
                e = frozen["entries"][f"{row}/{t}/s{s}"]
                dev = st.with_temperature(st.load_rows(ROOT / e["dev_dir"]), e["dev_temperature"])
                thr[f"{row}/s{s}"] = dev_threshold(dev)
                seeds.append(hy.run_rows(ROOT, e, "test"))
            named[row] = seeds
        aligned, _ = hy.align(named)
        aligned_all[t] = aligned
        thresholds[t] = thr
        per_task[t] = {}
        for row in ROWS:
            arrs = [st.arrays(r) for r in aligned[row]]
            cov = [fixed_coverage(x, thr[f"{row}/s{i}"]) for i, x in enumerate(arrs)]
            err = [fixed_error(x, thr[f"{row}/s{i}"]) for i, x in enumerate(arrs)]
            per_task[t][row] = {"coverage_per_seed": cov, "realized_error_per_seed": err,
                                "coverage": float(np.mean(cov)),
                                "realized_error": float(np.mean([e for e in err if e is not None])) if any(e is not None for e in err) else None,
                                "threshold_per_seed": [thr[f"{row}/s{i}"] for i in range(len(arrs))]}

    def stat_factory(tasks):
        def stat(draw):
            vals = []
            for t in tasks:
                k = np.mean([fixed_coverage(x, thresholds[t][f"kev4b-3k/s{i}"]) for i, x in enumerate(draw[t]["kev4b-3k"])])
                s = np.mean([fixed_coverage(x, thresholds[t][f"sft4b-3k/s{i}"]) for i, x in enumerate(draw[t]["sft4b-3k"])])
                vals.append(k - s)
            return float(np.mean(vals))
        return stat

    out = {"note": __doc__.split("\n\n")[0], "budget": BUDGET, "per_task": per_task, "thresholds": thresholds}
    for name, tasks in (("H3_devfixed", tuple(hy.TASKS)), ("H4b_devfixed", ("i1", "i2", "i3"))):
        draws = st.bootstrap({t: aligned_all[t] for t in tasks}, stat_factory(tasks), a.samples, a.seed)
        point = float(np.mean([per_task[t]["kev4b-3k"]["coverage"] - per_task[t]["sft4b-3k"]["coverage"] for t in tasks]))
        out[name] = {**hy.summarize(draws, point), "tasks": list(tasks), "passes": st.ci(draws)[0] > 0}
    (ROOT / a.out).write_text(json.dumps(out, indent=1) + "\n")
    for t in hy.TASKS:
        k, s = per_task[t]["kev4b-3k"], per_task[t]["sft4b-3k"]
        print(f"{t}: kev cov {k['coverage']:.3f} err {k['realized_error']} | sft cov {s['coverage']:.3f} err {s['realized_error']}")
    for name in ("H3_devfixed", "H4b_devfixed"):
        print(name, out[name]["point"], out[name]["ci95"], "passes" if out[name]["passes"] else "fails")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
