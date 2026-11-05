from conftest import REPO
from kevmed.tasks import i3_patho as i3

FX = REPO / "tests/fixtures/internal/raw"


def test_labels_and_time_split():
    by, consort = i3.build(FX, seed=1)
    allr = [r for v in by.values() for r in v]
    assert consort["reports"] == 16 and consort["dropped_no_behaviour"] == 2 and consort["dropped_short_text"] == 1
    assert all(r["_meta"]["year"] <= 2024 for r in by["train"]) and all(r["_meta"]["year"] == 2025 for r in by["dev"])
    r = next(x for x in allr if x["_meta"]["report_id"] == "R0000")
    assert r["questions"]["malignant"]["label"] is True and r["questions"]["metastatic"]["label"] is False
    r = next(x for x in allr if x["_meta"]["report_id"] == "R0002")
    assert r["questions"]["metastatic"]["label"] is True
    r = next(x for x in allr if x["_meta"]["report_id"] == "R0001")
    assert r["questions"]["malignant"]["label"] is False and "metastatic" not in r["questions"]


def test_mixed_codes_count_as_malignant_and_are_counted():
    by, consort = i3.build(FX, seed=1)
    allr = [r for v in by.values() for r in v]
    r3 = next(x for x in allr if x["_meta"]["report_id"] == "R0003")      # /2 + /3 -> malignant, mixed
    r4 = next(x for x in allr if x["_meta"]["report_id"] == "R0004")      # /3 + /6 -> malignant, metastatic omitted
    assert r3["questions"]["malignant"]["label"] is True and r4["questions"]["malignant"]["label"] is True
    assert "metastatic" not in r4["questions"] and consort["mixed_behaviour"] >= 2
