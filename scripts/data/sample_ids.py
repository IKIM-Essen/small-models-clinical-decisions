"""Write the I2 doc ids to OCR (the chapter-stratified sample drawn on the raw rows, before any OCR).
  uv run scripts/data/sample_ids.py --rows restricted/internal/raw/i2/rows.jsonl --out restricted/splits/i2-ids/doc_ids.txt --seed 20260925
Prints counts only."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kevmed import schema, splits
from kevmed.tasks import i2_letter


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--train", type=int, default=i2_letter.DEFAULT_SAMPLE["train"])
    ap.add_argument("--dev", type=int, default=i2_letter.DEFAULT_SAMPLE["dev"])
    ap.add_argument("--test", type=int, default=i2_letter.DEFAULT_SAMPLE["test"])
    ap.add_argument("--oversample", type=float, default=1.05)
    a = ap.parse_args(argv)
    if not splits.under_restricted_root(a.out):
        print(f"doc ids are row-level identifiers: --out must be under $KEV_RESTRICTED_ROOT ({splits.restricted_root()})", file=sys.stderr)
        return 4
    ids = i2_letter.select_sample(schema.read_jsonl(a.rows), a.seed, {"train": a.train, "dev": a.dev, "test": a.test}, a.oversample)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    flat = [d for k in ("train", "dev", "test") for d in ids[k]]
    a.out.write_text("\n".join(flat) + "\n")
    print({k: len(v) for k, v in ids.items()}, "total", len(flat))
    return 0


if __name__ == "__main__":
    sys.exit(main())
