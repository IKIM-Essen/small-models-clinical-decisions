"""How many I3 pathology reports state an ICD-O morphology code in their text, and
how often that code's behavior digit agrees with the malignancy label -> runs/i3-codes.json (counts only).

The I3 label is the ICD-O-3 behavior of the report's coded conclusion; the I3 builder, unlike I1 and I2, does not strip
codes from the report text, so part of the reports state the answer. Pattern: four digits starting with 8 or 9, a slash
and the behavior digit (e.g. 8140/3). Malignant per text = any behavior /3, /6 or /9, as in the label rule
(src/kevmed/icd.py). Reads the frozen split files on the pod; writes and prints counts only."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CODE = re.compile(r"(?<!\d)[89]\d{3}\s*/\s*(\d)(?!\d)")


def count(path: Path) -> dict:
    n = with_code = agree = 0
    for line in path.open():
        r = json.loads(line)
        n += 1
        behaviors = CODE.findall(r.get("state", "") if isinstance(r.get("state"), str) else json.dumps(r.get("state")))
        if not behaviors:
            continue
        with_code += 1
        label = r["questions"].get("malignant", {}).get("label")
        agree += (any(b in "369" for b in behaviors) == bool(label))
    return {"n_records": n, "with_code": with_code, "label_agrees": agree}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="runs/i3-codes.json")
    a = ap.parse_args(argv)
    root = Path(a.root)
    res = {s: count(root / "splits" / "i3" / f"{s}.jsonl") for s in ("train", "dev", "test")
           if (root / "splits" / "i3" / f"{s}.jsonl").exists()}
    (root / a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(f"wrote {a.out}: " + ", ".join(f"{s} {v['with_code']}/{v['n_records']}" for s, v in res.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
