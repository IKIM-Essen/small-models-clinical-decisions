# /// script
# requires-python = ">=3.11"
# ///
"""Apply a task's context strategy to a copy of a frozen split before kev sees it
.

I2 uses section-first truncation (kevmed.context.section_first): diagnosis and assessment sections first, then the
rest of the letter, cut to --budget tokens of the base model's tokenizer. The budget sits below kev's --max_state
because kev adds a few tokens of its own; kev.train reports any record that still does not fit.
The frozen split is never modified; the copy goes under data/derived/ (gitignored, same access rules as data/).
Prints counts only, never row content."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from kevmed import context
from kevmed.tasks.i2_letter import DIAGNOSIS_HEADINGS

# diagnosis sections (the I2 builder's own list) + assessment sections, in this order
I2_FIRST = ["hauptdiagnose", "diagnosen bei entlassung", "diagnosen", "diagnose", "nebendiagnosen",
            "beurteilung", "epikrise", "zusammenfassung"]
assert set(DIAGNOSIS_HEADINGS) <= set(I2_FIRST)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=7400)
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--base_revision", default="1001bb4d826a52d1f399e183466143f4da7b741b")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.base, revision=a.base_revision)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    n = cut = reordered_only = 0
    with open(a.data, encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            r = json.loads(line)
            text, was_cut = context.section_first(r["state"], I2_FIRST, a.budget, tok)
            cut += was_cut
            r = {**r, "state": text, "_meta": {**r["_meta"], "context": f"section_first:{a.budget}"}}
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"{a.data} -> {a.out}: {n} records, {cut} cut to {a.budget} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
