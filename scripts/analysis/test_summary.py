"""Runs/test-rows-summary.json and runs/frontier-test-summary.json from the scored test rows.

Each cell is [accuracy, macro-F1, automation at 5%]. Frozen rows: each run's dev-fitted temperature, seed means over the
frozen seeds (all three for the fine-tuned rows, or no cell), unpaired (per run, as kev.benchmark reports it). Frontier
rows: raw, as in H2; thinking modes get no automation, so their third
entry is null; so is frontier automation on I2 (22 options, top-20 log-probabilities). A row or cell that was not
run is null. Aggregates only; nothing row-level is written or printed."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hypotheses as hy  # noqa: E402
import stats as st  # noqa: E402

ROWS = ("kev4b-full", "kev4b-3k", "sft4b-3k", "classical-3k", "kev4b-zeroshot", "qwen4b-zeroshot-k0", "qwen4b-zeroshot-k3")
MULTI_SEED = {"kev4b-full", "kev4b-3k", "sft4b-3k", "classical-3k"}
# tasks with more than 20 options get accuracy and macro-F1 only on gateway models; the gateway returns
# the top 20 log-probabilities, so I2's 22 option letters can never all be scored.
NO_GATEWAY_AUTOMATION = {"i2"}


def cell(seed_runs: list, automation: bool = True) -> list:
    arrs = [st.arrays(r) for r in seed_runs]
    return [st.seed_mean(arrs, st.accuracy), st.seed_mean(arrs, st.macro_f1),
            st.seed_mean(arrs, st.automation) if automation else None]


def rows_summary(root: Path, frozen: dict, tasks=hy.TASKS) -> dict:
    out = {}
    for row in ROWS:
        seeds = hy.SEEDS if row in MULTI_SEED else (0,)
        out[row] = {}
        for t in tasks:
            runs = hy.gather(root, frozen, row, t, "test", seeds)
            out[row][t] = cell(runs) if runs else None
    return out


def frontier_summary(root: Path, tasks=hy.TASKS) -> dict:
    out = {}
    for model in hy.FRONTIER:
        for m in hy.FRONTIER_MODES:
            for t in tasks:
                rows = hy.frontier_rows(root, model, m, t)
                if rows is not None:
                    out[f"{model}/{m}/{t}"] = cell([rows], automation=m.startswith("nothink") and t not in NO_GATEWAY_AUTOMATION)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--rows_out", default="runs/test-rows-summary.json")
    ap.add_argument("--frontier_out", default="runs/frontier-test-summary.json")
    a = ap.parse_args(argv)
    root = Path(a.root)
    frozen = json.loads((root / "configs" / "frozen.json").read_text())
    rows, frontier = rows_summary(root, frozen), frontier_summary(root)
    (root / a.rows_out).write_text(json.dumps(rows, indent=1) + "\n")
    (root / a.frontier_out).write_text(json.dumps(frontier, indent=1) + "\n")
    print(f"wrote {a.rows_out} ({sum(v is not None for r in rows.values() for v in r.values())} cells) and "
          f"{a.frontier_out} ({len(frontier)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
