"""Outcome audit for I1 and I4: their labels are recorded outcomes (discharge time, next
admission), not judgements, so a clinician reading the admission state cannot check them. Instead this script takes the
label-check sample (same seed as `label_check.py`, so the clinicians' 10 rows are the first 10 of any larger sample),
looks every case up again in FMX by its encounter id through a separate per-case query (no date windows, no chunking,
no builder code), recomputes the label and prints counts only.

  uv run scripts/review/outcome_audit.py --task i1 --n 500        # runner Job (needs FMX_DSN and the split rows)
  uv run scripts/review/outcome_audit.py --task i4 --n 500

Disagreements can also come from FMX corrections after the extraction (2026-09-28); the report cannot tell the two
apart and says so."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from kevmed import schema
from kevmed.tasks import i1_los

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_check import sample  # noqa: E402  (same rows as the label check)

_FHIR_ID = re.compile(r"^[A-Za-z0-9\-.]{1,64}$")
WINDOW = timedelta(days=30)


def _ts(v) -> datetime | None:
    if v is None:
        return None
    return v if isinstance(v, datetime) else datetime.fromisoformat(str(v).replace("Z", "+00:00"))


def lookup_sql(case_ids: list[str]) -> str:
    """Admission, discharge and the next finished inpatient admission of the same patient, per encounter id."""
    bad = [c for c in case_ids if not _FHIR_ID.match(c)]
    if bad:
        raise ValueError(f"{len(bad)} case ids are not valid FHIR ids")
    ids = ", ".join(f"'{c}'" for c in case_ids)
    return f"""WITH c AS (
  SELECT e.id AS case_id, es._reference_id AS patient_rid, lower(ep.start) AS admit, lower(ep."end") AS discharge
  FROM encounter e
  JOIN encounter_period ep ON ep._resource = e._id
  LEFT JOIN encounter_subject es ON es._resource = e._id
  WHERE e.id IN ({ids})
)
SELECT c.case_id AS case_id, c.admit AS admit, c.discharge AS discharge,
  (SELECT MIN(lower(ep2.start))
     FROM encounter_subject es2
     JOIN encounter e2 ON e2._id = es2._resource AND e2.status = fmx_read_code('finished')
     JOIN encounter_period ep2 ON ep2._resource = e2._id
     JOIN encounter_class ec2 ON ec2._resource = e2._id AND ec2.code = fmx_read_code('IMP')
     JOIN encounter_type_coding etc2 ON etc2._resource = e2._id AND etc2.code = fmx_read_code('einrichtungskontakt')
    WHERE es2._reference_type = 'Patient' AND es2._reference_id = c.patient_rid
      AND lower(ep2.start) > c.discharge) AS next_admit
FROM c"""


def fmx_fetch(case_ids: list[str], batch: int = 200) -> dict[str, dict]:
    from kevmed.fmx import db
    out: dict[str, dict] = {}
    for i in range(0, len(case_ids), batch):
        for r in db.rows(lookup_sql(case_ids[i:i + batch])):
            out[r["case_id"]] = r
    return out


def _recompute(task: str, row: dict):
    admit, discharge = _ts(row["admit"]), _ts(row["discharge"])
    if discharge is None:
        return None
    if task == "i1":
        return i1_los.bucket(max(0, math.ceil((discharge - admit).total_seconds() / 86400)))
    nxt = _ts(row.get("next_admit"))
    return nxt is not None and nxt < discharge + WINDOW


def audit(task: str, records: list[dict], fetch) -> dict:
    qid = {"i1": "los", "i4": "readmit"}[task]
    found = fetch([r["_meta"]["case_id"] for r in records])
    confusion, agree, missing = Counter(), 0, 0
    for r in records:
        row = found.get(r["_meta"]["case_id"])
        got = _recompute(task, row) if row else None
        if got is None:
            missing += 1
            continue
        label = r["questions"][qid]["label"]
        agree += label == got
        confusion[f"label {label} / fmx {got}"] += 1
    return {"task": task, "sampled": len(records), "checked": len(records) - missing, "agree": agree,
            "disagree": len(records) - missing - agree, "missing_in_fmx": missing, "confusion": dict(sorted(confusion.items())),
            "note": "disagreement = builder error or an FMX correction after the 2026-09-28 extraction"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["i1", "i4"], required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--splits-dir", type=Path, default=Path("splits"))
    a = ap.parse_args(argv)
    records = sample(schema.read_jsonl(a.splits_dir / a.task / f"{a.split}.jsonl"), a.n, a.seed)
    print(json.dumps({**audit(a.task, records, fmx_fetch), "split": a.split, "seed": a.seed}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
