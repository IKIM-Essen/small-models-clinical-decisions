"""Per-seed test metrics of fine-tuned Kev-4B (3k) and LoRA-SFT (3k) -> runs/seed-spread-test.json.

On the pairs both models scored (as in H3), with each run's dev-fitted temperature: automation at 5%, ECE, Brier,
accuracy and macro-F1 for every seed, plus the min, max and standard deviation over the three seeds. Shows how much of a
Kev-4B minus LoRA-SFT difference is within seed-to-seed variation. Aggregates only; nothing row-level is written."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hypotheses as hy  # noqa: E402
import stats as st  # noqa: E402


def spread(root: Path, frozen: dict, tasks=hy.TASKS) -> dict:
    out = {}
    for t in tasks:
        kev, sft = hy.gather(root, frozen, "kev4b-3k", t, "test"), hy.gather(root, frozen, "sft4b-3k", t, "test")
        if kev is None or sft is None:
            continue
        aligned, _ = hy.align({"kev": kev, "sft": sft})
        out[t] = {}
        for side, runs in aligned.items():
            arrs = [st.arrays(r) for r in runs]
            per = {m: [float(fn(a)) for a in arrs] for m, fn in hy.METRICS.items()}
            out[t][side] = {m: {"seeds": v, "min": min(v), "max": max(v), "sd": float(np.std(v, ddof=1))} for m, v in per.items()}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="runs/seed-spread-test.json")
    a = ap.parse_args(argv)
    root = Path(a.root)
    res = spread(root, json.loads((root / "configs" / "frozen.json").read_text()))
    (root / a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(f"wrote {a.out}: {len(res)} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
