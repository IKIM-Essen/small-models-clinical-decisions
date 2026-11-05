import pytest

from conftest import REPO
from kevmed import splits
from kevmed.tasks import ed_state, p3_esi

FX = REPO / "tests/fixtures/physionet"


def test_load_stays_joins_and_counts():
    rows, counts = ed_state.load_stays(FX)
    assert counts["edstays"] == 60 and counts["stays_without_triage"] == 1 and counts["stays_without_patient"] == 1
    r = rows[0]
    assert set(r["state"]) == {"age", "sex", "arrival", "chief_complaint", "vitals", "pain"}
    assert "hadm_id" not in r["state"] and "disposition" not in r["state"]
    assert r["state"]["vitals"]["temperature_f"] in ("98.6", "not recorded")


def test_age_at_ed_visit():
    rows, _ = ed_state.load_stays(FX)
    by_stay = {r["stay_id"]: r for r in rows}
    assert by_stay["30000000"]["state"]["age"] == 22      # anchor 20 in 2148, visit 2150
    assert by_stay["30000056"]["state"]["age"] == 91      # capped


def test_p3_follows_the_patient_partition():
    p3, _ = p3_esi.build(FX, seed=1)
    part = ed_state.partition(FX, seed=1)
    for by in (p3,):
        for name, recs in by.items():
            assert all(part[r["_meta"]["subject_id"]] == name for r in recs)
        assert splits.overlap(by, "subject_id") == 0


def test_p3_scores_and_drops_missing_acuity():
    by, consort = p3_esi.build(FX, seed=1)
    assert consort["dropped_missing_acuity"] == 1 and consort["dropped_no_triage"] == 1
    r = next(iter(by["train"] + by["dev"] + by["test"]))
    q = r["questions"]["acuity"]
    assert q["type"] == "score" and len(q["criteria"]) == 5 and 0 <= q["label"] <= 4


def test_malformed_numeric_cells_raise_with_ids_only(tmp_path):
    import gzip
    import shutil

    from kevmed import physionet as pn
    root = tmp_path / "physionet"
    shutil.copytree(FX, root)
    p = root / pn.LAYOUT["ed_triage"]
    with gzip.open(p, "rt") as f:
        lines = f.read().splitlines()
    lines[1] = lines[1].rsplit(",", 2)[0] + ",SYNTHETIC-BAD," + lines[1].rsplit(",", 1)[1]   # acuity cell
    with gzip.open(p, "wt") as f:
        f.write("\n".join(lines) + "\n")
    with pytest.raises(ValueError) as e:
        ed_state.load_stays(root)
    assert "30000000" in str(e.value) and "SYNTHETIC" not in str(e.value)


def test_consort_reports_subject_overlap():
    _, c3 = p3_esi.build(FX, seed=1)
    assert c3["subjects_overlapping_splits"] == 0
