import gzip

from kevmed import physionet as pn


def test_layout_has_every_source():
    for key in ("mimic_cxr_reports", "mimic_cxr_chexpert", "mimic_cxr_split", "mednli", "ed_edstays", "ed_triage",
                "iv_patients", "iii_noteevents", "phenotypes"):
        assert key in pn.LAYOUT


def test_read_csv_gz_reads_plain_and_gzipped(tmp_path):
    (tmp_path / "a.csv").write_text("x,y\n1,2\n")
    with gzip.open(tmp_path / "b.csv.gz", "wt") as f:
        f.write("x,y\n3,4\n")
    assert list(pn.read_csv_gz(tmp_path / "a.csv")) == [{"x": "1", "y": "2"}]
    assert list(pn.read_csv_gz(tmp_path / "b.csv.gz")) == [{"x": "3", "y": "4"}]


def test_patient_partition_is_deterministic_and_order_independent():
    a = pn.patient_partition(["p3", "p1", "p2"] * 1 + [f"p{i}" for i in range(4, 200)], seed=1)
    b = pn.patient_partition([f"p{i}" for i in range(1, 200)], seed=1)
    assert a == b and set(a.values()) == {"train", "dev", "test"}
    assert pn.patient_partition([f"p{i}" for i in range(1, 200)], seed=2) != a


def test_age_at_uses_anchor_year_and_caps():
    assert pn.age_at(50, 2150, 2153) == 53
    assert pn.age_at(91, 2140, 2150) == 91
    assert pn.age_at(0, 2150, 2149) == 0
