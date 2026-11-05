# /// script
# requires-python = ">=3.11"
# ///
"""Strip `_meta` keys that collide with kev's reserved row-pairing fields before a record reaches
kev.benchmark --data.

P2's builder (src/kevmed/tasks/p2_mednli.py) tags every record with `_meta.pair_id` = MedNLI's own
`pairID` (provenance, duplicate of `_meta.group` which kevmed.splits actually uses for leakage-safe
grouping). kev.data.load_records merges a record's own `_meta` into the row kev.benchmark builds,
and kev/benchmark.py reads `meta.get("pair_id")`/`meta.get("sibling")` to reconstruct kev's own
contrastive a/b sibling pairs (kev/contrastive.py) for the `paired_flip` metric. Our `pair_id` is
not one of those pairs (no `sibling` field), so `paired_flip` finds an incomplete pair and raises,
crashing the whole report (scoring itself succeeds; only the aggregate summary crashes).

Frozen splits are never modified (Section 6.a/6.c immutability): this reads a split file and
writes a filtered copy elsewhere, for evaluation only. Only P2 is affected (checked: no other
task builder sets `_meta.pair_id`), but any future builder that reuses this field name would hit
the same crash, so the reasoning is written up here, not just worked around once.

Never prints row content: reads/writes JSONL, prints a count only.

Usage:
  uv run scripts/repro/kev_benchmark_data.py --data splits/p2/dev.jsonl \
      --out runs/pilot-kev08b-p2-eval/dev.sanitized.jsonl --strip-meta pair_id

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def strip_meta_keys(record: dict, keys: set[str]) -> dict:
    meta = record.get("_meta", {})
    if not (keys & meta.keys()):
        return record
    return {**record, "_meta": {k: v for k, v in meta.items() if k not in keys}}


def filter_jsonl(src: Path, dst: Path, strip_keys: set[str]) -> int:
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            record = strip_meta_keys(json.loads(line), strip_keys)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, type=Path, help="source JSONL (a frozen split file)")
    ap.add_argument("--out", required=True, type=Path, help="destination for the filtered copy")
    ap.add_argument("--strip-meta", required=True, help="comma-separated _meta keys to remove")
    a = ap.parse_args(argv)
    n = filter_jsonl(a.data, a.out, set(a.strip_meta.split(",")))
    print(f"wrote {n} records to {a.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
