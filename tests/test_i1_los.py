from conftest import REPO
from kevmed.tasks import i1_los as i1

FX = REPO / "tests/fixtures/internal/raw"


def test_invalid_and_open_periods_dropped():
    by, consort = i1.build(FX, seed=1)
    assert consort["cases"] == 24 and consort["dropped_open"] == 1 and consort["dropped_invalid_period"] == 1
    assert consort["dropped_no_auf"] == 2                      # no codes, and alpha-id only
    assert all(r["_meta"]["los_days"] >= 0 for v in by.values() for r in v)


def test_discharge_after_extraction_time_counts_as_open():
    by, consort = i1.build(FX, seed=1, extracted_at="2026-05-16T00:00:00+02:00")
    assert consort["dropped_open"] > 1


def test_buckets_state_and_year_split():
    by, consort = i1.build(FX, seed=1)
    allr = {r["_meta"]["case_id"]: r for v in by.values() for r in v}
    assert allr["C0000"]["questions"]["los"]["label"] == 0 and allr["C0001"]["questions"]["los"]["label"] == 1
    assert allr["C0003"]["questions"]["los"]["label"] == 2 and allr["C0004"]["questions"]["los"]["label"] == 3
    s = allr["C0000"]["state"]
    assert set(s) == {"age", "sex", "department", "admission_diagnoses", "first_labs", "prior_diagnoses"}
    assert "I21.0" not in s["admission_diagnoses"][0] and s["first_labs"][0]["unit"] == "mmol/L"
    assert all(r["_meta"]["year"] <= 2024 for r in by["train"]) and all(r["_meta"]["year"] == 2026 for r in by["test"])
