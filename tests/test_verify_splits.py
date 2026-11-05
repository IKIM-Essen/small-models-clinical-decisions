import hashlib

from conftest import load_script

vs = load_script("scripts/repro/verify_splits.py")


def _write(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def test_all_match_returns_no_problems(tmp_path):
    h1 = _write(tmp_path / "splits/p2/train.jsonl", '{"state": "secret row"}\n')
    h2 = _write(tmp_path / "splits/p2/test.jsonl", '{"state": "other row"}\n')
    (tmp_path / "splits/CHECKSUMS.sha256").write_text(f"{h1}  p2/train.jsonl\n{h2}  p2/test.jsonl\n")
    assert vs.verify(tmp_path / "splits", tmp_path / "splits/CHECKSUMS.sha256") == []


def test_mismatch_and_missing_are_reported(tmp_path):
    h1 = _write(tmp_path / "splits/a.jsonl", "x\n")
    (tmp_path / "splits/CHECKSUMS.sha256").write_text(f"{h1}  a.jsonl\n{'0' * 64}  b.jsonl\n")
    (tmp_path / "splits/a.jsonl").write_text("tampered\n")
    problems = vs.verify(tmp_path / "splits", tmp_path / "splits/CHECKSUMS.sha256")
    kinds = sorted(p["problem"] for p in problems)
    assert kinds == ["mismatch", "missing"]


def test_unlisted_split_file_is_reported(tmp_path):
    h1 = _write(tmp_path / "splits/a.jsonl", "x\n")
    _write(tmp_path / "splits/sneaky.jsonl", "y\n")
    (tmp_path / "splits/CHECKSUMS.sha256").write_text(f"{h1}  a.jsonl\n")
    problems = vs.verify(tmp_path / "splits", tmp_path / "splits/CHECKSUMS.sha256")
    assert [p["problem"] for p in problems] == ["unlisted"]


def test_cli_never_prints_row_content(tmp_path, capsys):
    h1 = _write(tmp_path / "splits/a.jsonl", "PATIENT_SECRET\n")
    (tmp_path / "splits/CHECKSUMS.sha256").write_text(f"{h1}  a.jsonl\n")
    (tmp_path / "splits/a.jsonl").write_text("PATIENT_SECRET_CHANGED\n")
    code = vs.main(["--splits", str(tmp_path / "splits")])
    out = capsys.readouterr()
    assert code == 1
    assert "PATIENT_SECRET" not in out.out + out.err


def test_write_mode_creates_checksum_file(tmp_path):
    _write(tmp_path / "splits/x/train.jsonl", "a\n")
    assert vs.main(["--splits", str(tmp_path / "splits"), "--write"]) == 0
    assert "x/train.jsonl" in (tmp_path / "splits/CHECKSUMS.sha256").read_text()
    assert vs.main(["--splits", str(tmp_path / "splits")]) == 0


def test_write_refuses_to_change_existing_hash(tmp_path):
    d = tmp_path / "splits"
    (d / "x").mkdir(parents=True)
    f = d / "x" / "train.jsonl"
    f.write_text("a\n")
    assert vs.main(["--splits", str(d), "--write"]) == 0
    f.write_text("b\n")
    assert vs.main(["--splits", str(d), "--write"]) == 3          # refused
    assert vs.main(["--splits", str(d)]) == 1                     # mismatch reported
    assert vs.main(["--splits", str(d), "--write", "--force"]) == 0
    assert vs.main(["--splits", str(d)]) == 0


def test_checksums_file_option_writes_elsewhere(tmp_path):
    d = tmp_path / "restricted" / "splits"
    (d / "p1").mkdir(parents=True)
    (d / "p1" / "dev.jsonl").write_text("r\n")
    out = tmp_path / "repo" / "splits" / "CHECKSUMS.restricted.sha256"
    out.parent.mkdir(parents=True)
    assert vs.main(["--splits", str(d), "--write", "--checksums-file", str(out)]) == 0
    assert "p1/dev.jsonl" in out.read_text() and not (d / "CHECKSUMS.sha256").exists()
    assert vs.main(["--splits", str(d), "--checksums-file", str(out)]) == 0


def test_only_row_files_are_checksummed(tmp_path):
    """Build.json and token_stats.json carry machine-dependent provenance (commit, source path) and live in git;
    the checksum file covers the row files only, which must reproduce byte for byte across machines."""
    d = tmp_path / "splits" / "s3"
    d.mkdir(parents=True)
    (d / "train.jsonl").write_text("row\n")
    (d / "build.json").write_text('{"git_commit": "abc"}')
    (d / "token_stats.json").write_text("{}")
    assert vs.main(["--splits", str(tmp_path / "splits"), "--write"]) == 0
    text = (tmp_path / "splits" / "CHECKSUMS.sha256").read_text()
    assert "s3/train.jsonl" in text and "build.json" not in text and "token_stats.json" not in text


def test_review_answers_are_not_split_files(tmp_path):
    """Splits/review/<task>/<reviewer>.jsonl holds label-check answers that grow while people review; freezing them
    (2026-09-29: two empty answer files ended up in the checksums) would make every later answer a hash mismatch."""
    _write(tmp_path / "splits" / "s3" / "dev.jsonl", "row\n")
    _write(tmp_path / "splits" / "review" / "s3" / "reviewer_a.jsonl", "")
    assert vs.main(["--splits", str(tmp_path / "splits"), "--write"]) == 0
    text = (tmp_path / "splits" / "CHECKSUMS.sha256").read_text()
    assert "s3/dev.jsonl" in text and "review" not in text
    (tmp_path / "splits" / "review" / "s3" / "reviewer_a.jsonl").write_text('{"id": "x"}\n')
    assert vs.main(["--splits", str(tmp_path / "splits")]) == 0
