"""I1 length-of-stay bucket at admission (Kev `score`), label from the Case encounter period, state from
admission-time information only: AUF diagnoses (ICD-10-GM display text, codes stripped),
department, age, sex, first 24 h labs, prior ICD-10-GM history. Year split train 2021 to 2024, dev 2025, test 2026.
Open cases: no discharge, or a discharge later than the extraction time (FMX stores an end on open cases)."""
from __future__ import annotations

import math
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path

from .. import icd, schema, splits

TASK, KEV_TYPE = "i1", "score"
BUCKETS = ["1 day or less", "2 to 3 days", "4 to 7 days", "more than 7 days"]
INSTRUCTIONS = "How long will this inpatient stay last, given the information available at admission?"
MAX_LABS, MAX_HISTORY = 30, 30


def bucket(los_days: int) -> int:
    return 0 if los_days <= 1 else 1 if los_days <= 3 else 2 if los_days <= 7 else 3


def split_of_year(year: int) -> str:
    return "train" if year <= 2024 else "dev" if year == 2025 else "test"


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def build(src: Path, seed: int, skip_bad: bool = False, extracted_at: str | None = None) -> tuple[dict[str, list[dict]], dict]:
    src = Path(src) / "i1"
    rows = schema.read_jsonl(src / "rows.jsonl")
    labs, hist = defaultdict(list), {}
    for l in schema.read_jsonl(src / "labs.jsonl"):
        labs[l["case_id"]].append(l)
    for h in schema.read_jsonl(src / "history.jsonl"):
        hist[h["case_id"]] = h
    cutoff = _ts(extracted_at) if extracted_at else None
    consort = OrderedDict(cases=len(rows), dropped_open=0, dropped_invalid_period=0, dropped_no_auf=0)
    by = {"train": [], "dev": [], "test": []}
    for r in rows:
        if not r.get("discharge"):
            consort["dropped_open"] += 1
            continue
        admit, discharge = _ts(r["admit"]), _ts(r["discharge"])
        if cutoff is not None and discharge > cutoff:
            consort["dropped_open"] += 1
            continue
        if discharge < admit:
            consort["dropped_invalid_period"] += 1
            continue
        if not r.get("auf_texts") or r.get("auf_alpha_only"):
            consort["dropped_no_auf"] += 1
            continue
        los_days = max(0, math.ceil((discharge - admit).total_seconds() / 86400))
        first = sorted(labs.get(r["case_id"], []), key=lambda l: l["taken"])[:MAX_LABS]
        state = {
            "age": r.get("age"), "sex": r.get("sex") or "unknown", "department": r.get("department") or "unknown",
            "admission_diagnoses": [icd.strip_icd_codes(t)[0].strip() for t in r["auf_texts"] if t],
            "first_labs": [{"name": l["name"], "value": l["value"], "unit": l.get("unit") or ""} for l in first],
            "prior_diagnoses": [icd.strip_icd_codes(t)[0].strip() for t in (hist.get(r["case_id"], {}).get("prior_texts") or [])[:MAX_HISTORY] if t],
        }
        year = int(r["year"])
        by[split_of_year(year)].append(schema.record(f"case-{r['case_id']}", state, {"los": schema.score(INSTRUCTIONS, BUCKETS, bucket(los_days))},
            {"case_id": r["case_id"], "patient_id": r["patient_id"], "group": r["patient_id"], "year": year, "los_days": los_days, "src": "medico-case"}))
    consort["patients_overlapping_splits"] = splits.overlap(by, "patient_id")
    for k in by:
        consort[f"n_{k}"] = len(by[k])
    return by, consort
