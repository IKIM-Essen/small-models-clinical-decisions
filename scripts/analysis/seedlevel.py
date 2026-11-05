# /// script
# requires-python = ">=3.11"
# ///
"""Sensitivity analysis: seed-level and patient-level inference for the H3/H4b
secondary metrics, Kev-4B 3k minus LoRA-SFT 3k on the test rows.

The primary bootstrap resamples records with the three seeds held fixed, so it does not carry seed variation. Here each
metric difference is computed per seed pair (seed s of Kev vs seed s of LoRA-SFT, aligned pairs) and per task; reported
are the three seed-level task means, their sign agreement, and a two-level bootstrap that resamples seed pairs (with
replacement from the three) and records (grouped by `_meta.group`, the patient or study, where the rows carry it)
together. Output: runs/seed-level-test.json. Declared after the primary results; never replaces them."""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import hypotheses as hy  # noqa: E402
import stats as st  # noqa: E402

METRICS = {"automation_at_5pct": st.automation, "ece": st.ece, "brier": st.brier, "accuracy": st.accuracy, "macro_f1": st.macro_f1}


def group_spans(rows):
    spans: "OrderedDict[str, list[int]]" = OrderedDict()
    for i, r in enumerate(rows):
        spans.setdefault(str(r.get("group") or r["id"]), []).append(i)
    return [np.array(v) for v in spans.values()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--samples", type=int, default=st.BOOTSTRAP_SAMPLES)
    ap.add_argument("--seed", type=int, default=st.BOOTSTRAP_SEED)
    ap.add_argument("--out", default="runs/seed-level-test.json")
    a = ap.parse_args(argv)
    frozen = json.loads((ROOT / "configs" / "frozen.json").read_text())
    aligned, spans, arrs = {}, {}, {}
    for t in hy.TASKS:
        named = {"kev": hy.gather(ROOT, frozen, "kev4b-3k", t, "test"), "sft": hy.gather(ROOT, frozen, "sft4b-3k", t, "test")}
        aligned[t], _ = hy.align(named)
        spans[t] = group_spans(aligned[t]["kev"][0])
        arrs[t] = {k: [st.arrays(r) for r in v] for k, v in aligned[t].items()}
    out = {"note": __doc__.split("\n\n")[0], "groups_per_task": {t: len(spans[t]) for t in hy.TASKS}, "per_seed": {}, "two_level": {}}
    for m, fn in METRICS.items():
        per_seed_task = {t: [float(fn(arrs[t]["kev"][s]) - fn(arrs[t]["sft"][s])) for s in range(3)] for t in hy.TASKS}
        seed_means = [float(np.mean([per_seed_task[t][s] for t in hy.TASKS])) for s in range(3)]
        internal = [float(np.mean([per_seed_task[t][s] for t in ("i1", "i2", "i3")])) for s in range(3)]
        out["per_seed"][m] = {"per_task": per_seed_task, "task_mean_per_seed": seed_means, "internal_mean_per_seed": internal,
                              "same_sign_all_seeds": bool(all(x < 0 for x in seed_means) or all(x > 0 for x in seed_means))}
    rng = np.random.default_rng(a.seed)
    for name, tasks in (("H3", tuple(hy.TASKS)), ("H4b", ("i1", "i2", "i3"))):
        draws = {m: np.empty(a.samples) for m in METRICS}
        for b in range(a.samples):
            seeds = rng.integers(0, 3, 3)                      # seed pairs with replacement
            idx = {t: np.concatenate([spans[t][p] for p in rng.integers(0, len(spans[t]), len(spans[t]))]) for t in tasks}
            for m, fn in METRICS.items():
                vals = []
                for t in tasks:
                    k = np.mean([fn(st.take(arrs[t]["kev"][s], idx[t])) for s in seeds])
                    f = np.mean([fn(st.take(arrs[t]["sft"][s], idx[t])) for s in seeds])
                    vals.append(k - f)
                draws[m][b] = np.mean(vals)
        out["two_level"][name] = {m: {"point": float(np.mean(out["per_seed"][m]["task_mean_per_seed" if name == "H3" else "internal_mean_per_seed"])),
                                      "ci95": list(st.ci(draws[m])),
                                      "p_kev_better": float(np.mean(draws[m] >= 0)) if m in ("ece", "brier") else st.p_greater_than_zero(draws[m])}
                                  for m in METRICS}
    (ROOT / a.out).write_text(json.dumps(out, indent=1) + "\n")
    for m in METRICS:
        print(m, "seed means", [round(x, 4) for x in out["per_seed"][m]["task_mean_per_seed"]], "same sign", out["per_seed"][m]["same_sign_all_seeds"],
              "H3 two-level", round(out["two_level"]["H3"][m]["point"], 4), [round(x, 4) for x in out["two_level"]["H3"][m]["ci95"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
