"""Majority-class accuracy and macro-F1 per task on test -> runs/majority-test.json.

For each question of a task, always predicting its most frequent training label; accuracy on the test split pooled over
the task's (record, question) rows. Uses only the label counts in splits/<task>/build.json (aggregates)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TASKS = ("p1", "p2", "p3", "i1", "i2", "i3")


def majority(build: dict) -> dict:
    train, test = build["splits"]["train"]["questions"], build["splits"]["test"]["questions"]
    hit = n = 0
    f1s = []
    for q, v in test.items():
        labels = v["labels"]
        top = max(train[q]["labels"], key=train[q]["labels"].get)
        k, nq = labels.get(top, 0), sum(labels.values())
        hit += k
        n += nq
        # macro-F1 of a constant prediction over the classes present in the test labels:
        # the predicted class has F1 2k / (nq + k) if present, every other class 0
        f1s.append((2 * k / (nq + k) if k else 0.0) / len(labels))
    return {"accuracy": hit / n, "macro_f1": sum(f1s) / len(f1s), "n_rows": n}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="runs/majority-test.json")
    a = ap.parse_args(argv)
    root = Path(a.root)
    res = {t: majority(json.loads((root / "splits" / t / "build.json").read_text()))
           for t in TASKS if (root / "splits" / t / "build.json").exists()}
    (root / a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(f"wrote {a.out}: {len(res)} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
