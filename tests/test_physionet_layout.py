"""Scripts/data/physionet_layout.py: arrange downloaded PhysioNet files into kevmed.physionet.LAYOUT (runner only)."""
import gzip
import subprocess
import sys
import zipfile

from conftest import REPO, load_script
from kevmed import physionet as pn

pl = load_script("scripts/data/physionet_layout.py")


def _incoming(tmp_path):
    inc = tmp_path / "incoming"
    inc.mkdir()
    for name in ("mli_train_v1.jsonl", "mli_dev_v1.jsonl", "mli_test_v1.jsonl"):
        (inc / name).write_text('{"pairID": "SYNTHETIC"}\n')
    for name in ("edstays.csv.gz", "triage.csv.gz", "patients.csv.gz", "NOTEEVENTS.csv.gz",
                 "mimic-cxr-2.0.0-split.csv.gz", "mimic-cxr-2.0.0-chexpert.csv.gz"):
        with gzip.open(inc / name, "wt") as f:
            f.write("a,b\nSYNTHETIC,1\n")
    with zipfile.ZipFile(inc / "mimic-cxr-reports.zip", "w") as z:
        z.writestr("files/p10/p10000001/s50000001.txt", "SYNTHETIC report")
        z.writestr("files/p11/p11000001/s51000001.txt", "SYNTHETIC report")
    with zipfile.ZipFile(inc / "phenotype-annotations-for-patient-notes-in-the-mimic-iii-database-1.20.3.zip", "w") as z:
        z.writestr("phenotype-annotations-for-patient-notes-in-the-mimic-iii-database-1.20.3/ACTdb102003.csv", "ROW_ID\nSYNTHETIC\n")
        z.writestr("phenotype-annotations-for-patient-notes-in-the-mimic-iii-database-1.20.3/LICENSE.txt", "x")
    return inc


def test_layout_places_every_source(tmp_path):
    inc = _incoming(tmp_path)
    root = tmp_path / "physionet"
    report = pl.arrange(inc, root)
    for key, rel in pn.LAYOUT.items():
        assert (root / rel).exists(), key
    assert (root / pn.LAYOUT["mimic_cxr_reports"] / "p10" / "p10000001" / "s50000001.txt").exists()
    assert report["mimic_cxr_reports"]["files"] == 2 and report["phenotypes"]["bytes"] > 0
    assert not any(inc.iterdir())                       # incoming emptied (moved, not copied)


def test_layout_refuses_root_outside_restricted_and_prints_no_text(tmp_path, capsys, monkeypatch):
    inc = _incoming(tmp_path)
    monkeypatch.setenv("KEV_RESTRICTED_ROOT", str(tmp_path / "restricted"))
    r = subprocess.run([sys.executable, str(REPO / "scripts/data/physionet_layout.py"), "--incoming", str(inc), "--root", str(tmp_path / "elsewhere")],
                       capture_output=True, text=True, env={"KEV_RESTRICTED_ROOT": str(tmp_path / "restricted"), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 4
    r = subprocess.run([sys.executable, str(REPO / "scripts/data/physionet_layout.py"), "--incoming", str(inc), "--root", str(tmp_path / "restricted" / "physionet")],
                       capture_output=True, text=True, env={"KEV_RESTRICTED_ROOT": str(tmp_path / "restricted"), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    assert "SYNTHETIC" not in r.stdout and "mednli" in r.stdout
