"""Triage-time state shared by P3 (acuity) and S1 (disposition), from MIMIC-IV-ED v2.2 + MIMIC-IV patients
: chief complaint, triage vitals, pain, arrival mode, age, sex. Nothing
post-decision (no hadm_id, diagnoses, medications). Never prints rows."""
from __future__ import annotations

from pathlib import Path

from .. import physionet

VITALS = [("temperature", "temperature_f"), ("heartrate", "heart_rate"), ("resprate", "respiratory_rate"),
          ("o2sat", "o2_saturation"), ("sbp", "systolic_bp"), ("dbp", "diastolic_bp")]
NOT_RECORDED = "not recorded"


def state_of(stay: dict, triage: dict, age: int) -> dict:
    return {
        "age": age,
        "sex": stay.get("gender") or NOT_RECORDED,
        "arrival": stay.get("arrival_transport") or NOT_RECORDED,
        "chief_complaint": (triage.get("chiefcomplaint") or "").strip() or NOT_RECORDED,
        "vitals": {name: (triage.get(col) or "").strip() or NOT_RECORDED for col, name in VITALS},
        "pain": (triage.get("pain") or "").strip() or NOT_RECORDED,
    }


def load_stays(src: Path) -> tuple[list[dict], dict]:
    src = Path(src)
    triage = {r["stay_id"]: r for r in physionet.read_csv_gz(src / physionet.LAYOUT["ed_triage"])}
    patients = {r["subject_id"]: r for r in physionet.read_csv_gz(src / physionet.LAYOUT["iv_patients"])}
    counts = {"edstays": 0, "triage_rows": len(triage), "stays_without_triage": 0, "stays_without_patient": 0}
    rows = []
    for stay in physionet.read_csv_gz(src / physionet.LAYOUT["ed_edstays"]):
        counts["edstays"] += 1
        t = triage.get(stay["stay_id"])
        p = patients.get(stay["subject_id"])
        if t is None:
            counts["stays_without_triage"] += 1
            continue
        if p is None:
            counts["stays_without_patient"] += 1
            continue
        acuity = (t.get("acuity") or "").strip()
        try:
            year = int(stay["intime"][:4])
            acuity_level = int(float(acuity)) if acuity else None
        except ValueError:
            raise ValueError(f"stay {stay['stay_id']}: unparseable intime or acuity") from None
        try:
            age = physionet.age_at(int(p["anchor_age"]), int(p["anchor_year"]), year)
        except ValueError:
            raise ValueError(f"subject {stay['subject_id']}: unparseable anchor fields") from None
        rows.append({"stay_id": stay["stay_id"], "subject_id": stay["subject_id"], "disposition": stay.get("disposition", ""),
                     "acuity": acuity_level, "state": state_of(stay, t, age)})
    return rows, counts


def partition(src: Path, seed: int) -> dict[str, str]:
    ids = [r["subject_id"] for r in physionet.read_csv_gz(Path(src) / physionet.LAYOUT["ed_edstays"])]
    return physionet.patient_partition(ids, seed)
