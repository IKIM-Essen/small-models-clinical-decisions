"""Scripts/data/sample_ids.py: the I2 doc ids to OCR, drawn from rows.jsonl with the split seed before any OCR."""
import subprocess
import sys

from conftest import REPO
from kevmed.tasks import i2_letter as i2

FX = REPO / "tests/fixtures/internal/raw"


def test_select_sample_draws_per_split_with_oversampling():
    rows = i2.schema.read_jsonl(FX / "i2" / "rows.jsonl")
    ids = i2.select_sample(rows, seed=20260925, sample={"train": 4, "dev": 2, "test": 2}, oversample=1.5)
    assert set(ids) == {"train", "dev", "test"} and len(ids["train"]) == 6 and len(ids["dev"]) == 3 and len(ids["test"]) == 3
    assert "D0100" in ids["train"] and "D0000" not in ids["train"]           # latest letter per case, unmapped code excluded
    assert i2.select_sample(rows, seed=20260925, sample={"train": 4, "dev": 2, "test": 2}, oversample=1.5) == ids


def test_cli_writes_doc_ids_and_counts_only(tmp_path):
    import os
    out = tmp_path / "doc_ids.txt"
    r = subprocess.run([sys.executable, str(REPO / "scripts/data/sample_ids.py"), "--rows", str(FX / "i2" / "rows.jsonl"),
                        "--out", str(out), "--seed", "1", "--train", "4", "--dev", "2", "--test", "2", "--oversample", "1.5"],
                       capture_output=True, text=True, cwd=REPO, env={**os.environ, "KEV_RESTRICTED_ROOT": str(tmp_path)})
    assert r.returncode == 0, r.stderr
    lines = out.read_text().splitlines()
    assert len(lines) == len(set(lines)) == 12 and all(l.startswith("D") for l in lines)
    assert "SYNTHETIC" not in r.stdout and "12" in r.stdout


def test_cli_refuses_out_outside_restricted_root(tmp_path):
    import os
    env = {**os.environ, "KEV_RESTRICTED_ROOT": str(tmp_path / "restricted")}
    r = subprocess.run([sys.executable, str(REPO / "scripts/data/sample_ids.py"), "--rows", str(FX / "i2" / "rows.jsonl"),
                        "--out", str(tmp_path / "shared" / "doc_ids.txt"), "--seed", "1"], capture_output=True, text=True, cwd=REPO, env=env)
    assert r.returncode == 4 and "restricted" in r.stderr
