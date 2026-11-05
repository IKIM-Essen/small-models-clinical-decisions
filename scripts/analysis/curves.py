"""Automation at error budgets 1% to 20% per row and task on test -> runs/curves-test.json.

Coverage at each budget is kev's own coverage_at_error (scripts/analysis/stats.py automation), with each run's dev-fitted
temperature applied and seed means over the frozen seeds, as in the hypotheses. Kev-4B 3k and LoRA-SFT 3k are restricted
to the pairs both scored (as in H3), so their 5% points equal the H3 per-task numbers. Frontier rows (test only, no dev
fit) are the two no-thinking modes of each model. Writes aggregates only; the figure scripts never read rows."""
from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hypotheses as hy  # noqa: E402
import stats as st  # noqa: E402

BUDGETS = [round(0.01 * i, 2) for i in range(1, 21)]
ROWS = ("kev4b-3k", "sft4b-3k", "kev4b-full", "kev4b-zeroshot", "qwen4b-zeroshot-k0")
NO_THINKING = ("nothink-k0", "nothink-k3")


def curve(seed_runs: list) -> list[float]:
    arrs = [st.arrays(r) for r in seed_runs]
    return [st.seed_mean(arrs, functools.partial(st.automation, budget=b)) for b in BUDGETS]


MULTI_SEED = {"kev4b-3k", "sft4b-3k", "kev4b-full"}   # all three seeds or no curve, as in the hypotheses


def build(root: Path, frozen: dict, tasks=hy.TASKS) -> dict:
    out = {"split": "test", "budgets": BUDGETS, "rows": {r: {} for r in ROWS}, "frontier": {}}
    for t in tasks:
        runs = {r: hy.gather(root, frozen, r, t, "test", hy.SEEDS if r in MULTI_SEED else (0,)) for r in ROWS}
        if runs["kev4b-3k"] and runs["sft4b-3k"]:
            aligned, _ = hy.align({"kev4b-3k": runs["kev4b-3k"], "sft4b-3k": runs["sft4b-3k"]})
            runs.update(aligned)
        for r, seed_runs in runs.items():
            if seed_runs:                                   # None: row (or one of its seeds) not frozen
                out["rows"][r][t] = curve(seed_runs)
        for model in hy.FRONTIER:
            for m in NO_THINKING:
                if t == "i2":                                # no gateway automation for > 20 options
                    continue
                rows = hy.frontier_rows(root, model, m, t)
                if rows is not None:
                    out["frontier"][f"{model}/{m}/{t}"] = curve([rows])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="runs/curves-test.json")
    a = ap.parse_args(argv)
    root = Path(a.root)
    res = build(root, json.loads((root / "configs" / "frozen.json").read_text()))
    (root / a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(f"wrote {a.out}: {sum(len(v) for v in res['rows'].values())} row curves, {len(res['frontier'])} frontier curves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
