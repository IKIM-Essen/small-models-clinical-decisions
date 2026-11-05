import pytest

from conftest import load_script
from kevmed import schema
from kevmed.tasks import i1_los

oa = load_script("scripts/review/outcome_audit.py")


def _i1(case, bucket_label):
    return schema.record(f"case-{case}", {"age": 1}, {"los": schema.score(i1_los.INSTRUCTIONS, i1_los.BUCKETS, bucket_label)},
                         {"case_id": case, "patient_id": "p", "los_days": 0})


def _i4(case, label):
    return schema.record(f"case-{case}", {"age": 1}, {"readmit": schema.noul("Readmitted?", label)}, {"case_id": case, "patient_id": "p"})


def test_i1_recomputes_the_bucket_from_the_encounter_period():
    recs = [_i1("a", 1), _i1("b", 2), _i1("c", 0), _i1("gone", 3)]
    fmx = {"a": {"admit": "2025-03-01T10:00:00+00:00", "discharge": "2025-03-03T09:00:00+00:00", "next_admit": None},   # 2 days -> 1
           "b": {"admit": "2025-03-01T10:00:00+00:00", "discharge": "2025-03-02T09:00:00+00:00", "next_admit": None},   # 1 day -> 0
           "c": {"admit": "2025-03-01T10:00:00+00:00", "discharge": "2025-03-01T18:00:00+00:00", "next_admit": None}}   # 0 -> 0
    out = oa.audit("i1", recs, lambda ids: {k: v for k, v in fmx.items() if k in ids})
    assert (out["checked"], out["agree"], out["disagree"], out["missing_in_fmx"]) == (3, 2, 1, 1)
    assert out["confusion"] == {"label 0 / fmx 0": 1, "label 1 / fmx 1": 1, "label 2 / fmx 0": 1}


def test_i4_recomputes_readmission_within_30_days():
    recs = [_i4("a", True), _i4("b", False), _i4("c", False), _i4("d", True)]
    fmx = {"a": {"admit": "2025-01-01T00:00:00+00:00", "discharge": "2025-01-05T00:00:00+00:00", "next_admit": "2025-01-20T00:00:00+00:00"},
           "b": {"admit": "2025-01-01T00:00:00+00:00", "discharge": "2025-01-05T00:00:00+00:00", "next_admit": "2025-03-01T00:00:00+00:00"},
           "c": {"admit": "2025-01-01T00:00:00+00:00", "discharge": "2025-01-05T00:00:00+00:00", "next_admit": "2025-01-10T00:00:00+00:00"},
           "d": {"admit": "2025-01-01T00:00:00+00:00", "discharge": "2025-01-05T00:00:00+00:00", "next_admit": None}}
    out = oa.audit("i4", recs, lambda ids: fmx)
    assert (out["checked"], out["agree"], out["disagree"]) == (4, 2, 2)
    assert out["confusion"] == {"label False / fmx False": 1, "label False / fmx True": 1, "label True / fmx False": 1, "label True / fmx True": 1}


def test_lookup_sql_quotes_only_valid_fhir_ids():
    sql = oa.lookup_sql(["abc-1.2", "X9"])
    assert "'abc-1.2'" in sql and "'X9'" in sql and "einrichtungskontakt" in sql
    with pytest.raises(ValueError):
        oa.lookup_sql(["a'; DROP TABLE x; --"])


def test_report_holds_counts_only():
    recs = [_i1("secret-case-id", 1)]
    fmx = {"secret-case-id": {"admit": "2025-03-01T10:00:00+00:00", "discharge": "2025-03-03T09:00:00+00:00", "next_admit": None}}
    out = oa.audit("i1", recs, lambda ids: fmx)
    assert "secret-case-id" not in repr(out)
