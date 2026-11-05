"""P3 triage acuity (ESI 1 to 5) from MIMIC-IV-ED triage as a Kev `score`.
Ordinal, noisy label (moderate inter-rater agreement; ceiling stated in Methods)."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from .. import schema, splits
from . import ed_state

TASK, KEV_TYPE = "p3", "score"
LEVELS = ["ESI 1: resuscitation, immediate life-saving intervention", "ESI 2: emergent, high risk",
          "ESI 3: urgent, two or more resources expected", "ESI 4: less urgent, one resource expected",
          "ESI 5: non-urgent, no resources expected"]
INSTRUCTIONS = "How acute is this patient at triage on the Emergency Severity Index?"


def build(src: Path, seed: int, skip_bad: bool = False) -> tuple[dict[str, list[dict]], dict]:
    rows, counts = ed_state.load_stays(src)
    consort = OrderedDict(edstays=counts["edstays"], dropped_no_triage=counts["stays_without_triage"],
                          dropped_no_patient=counts["stays_without_patient"], dropped_missing_acuity=0, dropped_acuity_out_of_range=0)
    records = []
    for r in rows:
        if r["acuity"] is None:
            consort["dropped_missing_acuity"] += 1
            continue
        if not 1 <= r["acuity"] <= 5:
            consort["dropped_acuity_out_of_range"] += 1
            continue
        records.append(schema.record(f"ed-{r['stay_id']}", r["state"], {"acuity": schema.score(INSTRUCTIONS, LEVELS, r["acuity"] - 1)},
                                     {"stay_id": r["stay_id"], "subject_id": r["subject_id"], "group": r["subject_id"], "src": "mimic-iv-ed"}))
    part = ed_state.partition(src, seed)
    by = splits.assign(records, {rec["_meta"]["id"]: part[rec["_meta"]["subject_id"]] for rec in records})
    by = {k: by.get(k, []) for k in ("train", "dev", "test")}
    consort["records"] = len(records)
    consort["subjects_overlapping_splits"] = splits.overlap(by, "subject_id")
    for k in by:
        consort[f"n_{k}"] = len(by[k])
    return by, consort
