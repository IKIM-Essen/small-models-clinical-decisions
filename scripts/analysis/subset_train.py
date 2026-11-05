# /// script
# requires-python = ">=3.11"
# ///
"""Seeded, group-preserving training subset of at most n records (H2 "<= 3k examples per task",
H3 "identical data" for Kev and the LoRA-SFT baseline).

Groups (`_meta.group`: patient, MIMIC subject, MedNLI pair) are shuffled with the seed and taken whole while
they fit, so no group is split and the subset never exceeds n. Records keep the source file's order.
The frozen split is only read; the subset goes under data/derived/ (gitignored). Prints and writes counts
and hashes only, never row content."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, OrderedDict
from pathlib import Path


def sample_by_group(records: list[dict], n: int, seed: int) -> list[dict]:
    groups: "OrderedDict[str, list[int]]" = OrderedDict()
    for i, r in enumerate(records):
        groups.setdefault(r["_meta"]["group"], []).append(i)
    order = list(groups)
    random.Random(seed).shuffle(order)
    keep: set[int] = set()
    for g in order:
        if len(keep) + len(groups[g]) <= n:
            keep.update(groups[g])
        if len(keep) == n:
            break
    return [r for i, r in enumerate(records) if i in keep]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20261017)
    a = ap.parse_args(argv)
    src, out, summ = Path(a.data), Path(a.out), Path(a.summary)
    if out.exists():
        print(f"{out} exists: refusing to overwrite a subset other runs may already use (delete it on purpose)")
        return 1
    records = [json.loads(line) for line in src.open(encoding="utf-8") if line.strip()]
    picked = sample_by_group(records, a.n, a.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    labels: dict[str, Counter] = {}
    for r in picked:
        for qid, q in r["questions"].items():
            labels.setdefault(qid, Counter())[str(q["label"])] += 1
    summary = {"source": str(src), "source_sha256": _sha256(src), "out": str(out), "out_sha256": _sha256(out),
               "n_requested": a.n, "n_records": len(picked), "n_groups": len({r["_meta"]["group"] for r in picked}),
               "n_source_records": len(records), "seed": a.seed,
               "labels": {q: dict(sorted(c.items())) for q, c in sorted(labels.items())}}
    summ.parent.mkdir(parents=True, exist_ok=True)
    summ.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"{src}: {len(picked)} of {len(records)} records ({summary['n_groups']} groups) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
