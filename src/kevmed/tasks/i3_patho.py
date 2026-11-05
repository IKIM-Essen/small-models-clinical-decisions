"""I3 ISP pathology report -> malignant or not (Kev `noul`), label from the pathologist's structured ICD-O-3
conclusionCode. Coded subset only (selection bias
stated in Methods). Time split: train <= 2024, dev 2025, test 2026. Reads raw JSONL written by fmx_extract.py."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from .. import icd, schema, splits

TASK, KEV_TYPE = "i3", "noul"
Q_MALIGNANT = "Does this pathology report describe a malignant neoplasm?"
Q_METASTATIC = "Is the malignant tumour in this report a metastasis rather than a primary tumour?"
MIN_CHARS = 200


def split_of_year(year: int) -> str:
    return "train" if year <= 2024 else "dev" if year == 2025 else "test"


def build(src: Path, seed: int, skip_bad: bool = False) -> tuple[dict[str, list[dict]], dict]:
    rows = schema.read_jsonl(Path(src) / "i3" / "rows.jsonl")
    consort = OrderedDict(reports=len(rows), dropped_no_behaviour=0, dropped_short_text=0, mixed_behaviour=0)
    by = {"train": [], "dev": [], "test": []}
    for r in rows:
        behaviours = [b for b in (icd.icdo3_behaviour(c) for c in r.get("icdo3_codes") or []) if b is not None]
        if not behaviours:
            consort["dropped_no_behaviour"] += 1
            continue
        text = (r.get("markdown") or "").strip()
        if len(text) < MIN_CHARS:
            consort["dropped_short_text"] += 1
            continue
        malignant = any(icd.is_malignant(b) for b in behaviours)
        if len(set(behaviours)) > 1:
            consort["mixed_behaviour"] += 1
        qs = {"malignant": schema.noul(Q_MALIGNANT, malignant)}
        mal = {b for b in behaviours if icd.is_malignant(b)}
        if mal and mal <= {3, 6} and len(mal) == 1:
            qs["metastatic"] = schema.noul(Q_METASTATIC, mal == {6})
        year = int(r["year"])
        by[split_of_year(year)].append(schema.record(f"isp-{r['report_id']}", text, qs,
            {"report_id": r["report_id"], "patient_id": r["patient_id"], "group": r["patient_id"], "year": year, "src": "isp-pathology"}))
    consort["patients_overlapping_splits"] = splits.overlap(by, "patient_id")
    for k in by:
        consort[f"n_{k}"] = len(by[k])
    return by, consort
