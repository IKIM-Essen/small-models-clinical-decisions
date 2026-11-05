# /// script
# requires-python = ">=3.11"
# ///
"""Sensitivity analysis: I3 without code-bearing reports and without patients seen in
training. Writes data/derived/i3-3k-nocode/{train,dev,test}.jsonl: train = the frozen 3k subset minus reports whose text
states an ICD-O morphology code (scripts/analysis/i3_codes.py pattern); dev and test = the frozen splits minus code-bearing
reports and minus every patient (`_meta.patient_id`) that occurs in the filtered training file. Counts only are printed
and written to data/derived/i3-3k-nocode/counts.json; never prints row content."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE = re.compile(r"(?<!\d)[89]\d{3}\s*/\s*(\d)(?!\d)")
OUT = ROOT / "data" / "derived" / "i3-3k-nocode"


def has_code(r):
    s = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"])
    return bool(CODE.search(s))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {}
    train = [json.loads(l) for l in open(ROOT / "data/derived/i3-3k/train.jsonl") if l.strip()]
    kept = [r for r in train if not has_code(r)]
    patients = {r["_meta"]["patient_id"] for r in kept}
    counts["train"] = {"n": len(train), "code_bearing": len(train) - len(kept), "kept": len(kept), "patients": len(patients)}
    with open(OUT / "train.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for split in ("dev", "test"):
        rows = [json.loads(l) for l in open(ROOT / f"splits/i3/{split}.jsonl") if l.strip()]
        no_code = [r for r in rows if not has_code(r)]
        unseen = [r for r in no_code if r["_meta"]["patient_id"] not in patients]
        counts[split] = {"n": len(rows), "code_bearing": len(rows) - len(no_code), "seen_patient": len(no_code) - len(unseen), "kept": len(unseen)}
        with open(OUT / f"{split}.jsonl", "w") as f:
            for r in unseen:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "counts.json").write_text(json.dumps(counts, indent=1) + "\n")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
