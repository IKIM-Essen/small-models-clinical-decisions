from conftest import REPO
from kevmed.tasks import p1_cxr as p1

FX = REPO / "tests/fixtures/physionet"


def test_chexpert_value_coding():
    by, consort = p1.build(FX, seed=1)
    r = next(x for split in by.values() for x in split if x["_meta"]["study_id"] == "50000001")
    q = r["questions"]
    assert q["atelectasis"]["label"] is True and q["cardiomegaly"]["label"] is False
    assert q["consolidation"]["label"] is False            # blank = not mentioned = absent
    assert "edema" not in q                                 # -1.0 = uncertain = masked
    assert q["no_finding"]["label"] is False and len(q) == 13


def test_official_split_and_drops():
    by, consort = p1.build(FX, seed=1)
    assert consort["studies_with_report"] == 7 and consort["dropped_no_chexpert_row"] == 1 and consort["dropped_all_uncertain"] == 1
    ids = {k: {r["_meta"]["study_id"] for r in v} for k, v in by.items()}
    assert ids["train"] == {"50000001", "50000002"} and ids["dev"] == {"50000003", "50000004"} and ids["test"] == {"50000007"}
    assert consort["subjects_overlapping_splits"] == 0
    r = by["dev"][0]
    assert "SYNTHETIC findings" in r["state"] and "\n\n\n" not in r["state"] and r["_meta"]["group"] == "10000002"


def test_report_path_rule():
    assert p1.report_path("10000032", "50414267").as_posix().endswith("files/p10/p10000032/s50414267.txt")


def test_unexpected_chexpert_values_are_counted_or_raise(tmp_path):
    import gzip
    import shutil

    import pytest

    from kevmed import physionet as pn
    root = tmp_path / "physionet"
    shutil.copytree(FX, root)
    p = root / pn.LAYOUT["mimic_cxr_chexpert"]
    with gzip.open(p, "rt") as f:
        lines = f.read().splitlines()
    lines[1] = lines[1].replace("1.0", "1").replace("0.0", "nan", 1)      # integer-style true and a nan
    with gzip.open(p, "wt") as f:
        f.write("\n".join(lines) + "\n")
    with pytest.raises(ValueError) as e:
        p1.build(root, seed=1)
    assert "50000001" in str(e.value) and "SYNTHETIC" not in str(e.value)
    by, consort = p1.build(root, seed=1, skip_bad=True)
    assert consort["unexpected_chexpert_values"] == 1 and consort["subjects_overlapping_splits"] == 0
    r = next(x for split in by.values() for x in split if x["_meta"]["study_id"] == "50000001")
    assert r["questions"]["atelectasis"]["label"] is True                   # "1" parsed as 1.0
