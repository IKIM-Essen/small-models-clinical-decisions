# /// script
# requires-python = ">=3.11"
# ///
"""Post hoc analysis: reliability-diagram bins.

Per task, test rows of Kev-4B 3k and LoRA-SFT 3k (the three seeds pooled, each at its dev-fitted temperature, as in
scripts/analysis/hypotheses.py) and the best non-thinking frontier row (the model and mode with the highest test macro-F1 among
the non-thinking modes; thinking rows carry no usable confidence). Ten equal-width bins of the top-option confidence on
[0, 1]: count, mean confidence and accuracy per bin. Counts only. Output: runs/reliability.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import hypotheses as hy  # noqa: E402
import stats as st  # noqa: E402

BINS = 10


def bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = BINS) -> list[dict]:
    """Equal-width bins on [0, 1]; the last bin is closed on the right. Empty bins carry n = 0 and no means."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = which == b
        n = int(m.sum())
        out.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": n,
                    "confidence": float(conf[m].mean()) if n else None, "accuracy": float(correct[m].mean()) if n else None})
    return out


def ece_from_bins(bs: list[dict]) -> float:
    total = sum(b["n"] for b in bs)
    return float(sum(b["n"] * abs(b["confidence"] - b["accuracy"]) for b in bs if b["n"]) / total)


def pooled(seed_rows: list[list[dict]]) -> tuple[np.ndarray, np.ndarray]:
    arrs = [st.arrays(r) for r in seed_rows]
    return np.concatenate([a.conf for a in arrs]), np.concatenate([a.correct for a in arrs])


def main():
    frozen = json.loads((ROOT / "configs" / "frozen.json").read_text())
    h2 = json.loads((ROOT / "runs" / "hypotheses-test.json").read_text())["H2"]["models"]
    long = {v: k for k, v in hy.FRONTIER.items()}
    out = {"note": __doc__.split("\n\n")[0], "bins": BINS, "tasks": {}}
    for t in hy.TASKS:
        cand = [(f1, short, m) for short in h2 for m, f1 in h2[short]["per_task"][t]["frontier_all_modes"].items() if m.startswith("nothink")]
        f1, short, mode = max(cand)
        rows = {"kev": hy.gather(ROOT, frozen, "kev4b-3k", t, "test"), "sft": hy.gather(ROOT, frozen, "sft4b-3k", t, "test"),
                "frontier": [hy.frontier_rows(ROOT, long[short], mode, t)]}
        out["tasks"][t] = {"frontier_row": {"model": long[short], "mode": mode}}
        for name, seeds in rows.items():
            conf, correct = pooled(seeds)
            bs = bins(conf, correct)
            out["tasks"][t][name] = {"n": int(len(conf)), "ece_10bin": ece_from_bins(bs), "bins": bs}
        print(t, {k: round(v["ece_10bin"], 4) for k, v in out["tasks"][t].items() if "ece_10bin" in v}, short, mode, flush=True)
    (ROOT / "runs" / "reliability.json").write_text(json.dumps(out, indent=1) + "\n")
    print("wrote runs/reliability.json")


if __name__ == "__main__":
    main()
