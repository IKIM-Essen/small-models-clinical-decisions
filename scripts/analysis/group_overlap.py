"""Patients (record groups) shared across splits, per task -> runs/group-overlap.json (aggregate counts only).

Counts the `_meta.group` ids (patient or subject ids, as each builder sets them) that occur in more than one of train,
dev, test and the 3k subset. This script writes them to a committed file. Reads the frozen split files on the pod; writes and prints counts only."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

TASKS = ("p1", "p2", "p3", "i1", "i2", "i3")


def groups(path: Path) -> set[str]:
    out = set()
    with path.open() as f:
        for line in f:
            g = json.loads(line).get("_meta", {}).get("group")
            if g is not None:
                out.add(str(g))
    return out


def overlap(root: Path, task: str) -> dict | None:
    files = {s: root / "splits" / task / f"{s}.jsonl" for s in ("train", "dev", "test")}
    sub = root / "data" / "derived" / f"{task}-3k" / "train.jsonl"
    if sub.exists():
        files["subset"] = sub
    if not all(files[s].exists() for s in ("train", "dev", "test")):
        return None
    g = {s: groups(p) for s, p in files.items()}
    out = {f"n_{s}": len(v) for s, v in g.items()}
    out.update({f"{a}_{b}": len(g[a] & g[b]) for a, b in combinations(g, 2)})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="runs/group-overlap.json")
    a = ap.parse_args(argv)
    root = Path(a.root)
    res = {t: o for t in TASKS if (o := overlap(root, t)) is not None}
    (root / a.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(f"wrote {a.out}: {len(res)} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
