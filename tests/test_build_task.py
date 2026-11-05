import json
import os
import subprocess
import sys

from conftest import REPO


def _p2_args(tmp_path):
    return [sys.executable, str(REPO / "scripts/data/build_task.py"), "--task", "p2", "--src",
            str(REPO / "tests/fixtures/physionet"), "--out", str(tmp_path / "splits" / "p2"), "--seed", "1", "--skip-bad"]


def test_cli_builds_fixture_and_prints_counts_only(tmp_path):
    env = {**os.environ, "KEV_RESTRICTED_ROOT": str(tmp_path)}
    r = subprocess.run(_p2_args(tmp_path), capture_output=True, text=True, cwd=REPO, env=env)
    assert r.returncode == 0, r.stderr
    assert "SYNTHETIC" not in r.stdout
    b = json.loads((tmp_path / "splits" / "p2" / "build.json").read_text())
    assert b["task"] == "p2" and b["seed"] == 1
    assert (tmp_path / "splits" / "p2" / "test.jsonl").exists()


def test_cli_refuses_to_overwrite(tmp_path):
    env = {**os.environ, "KEV_RESTRICTED_ROOT": str(tmp_path)}
    args = _p2_args(tmp_path)
    assert subprocess.run(args, capture_output=True, text=True, cwd=REPO, env=env).returncode == 0
    r = subprocess.run(args, capture_output=True, text=True, cwd=REPO, env=env)
    assert r.returncode != 0 and "immutable" in (r.stderr + r.stdout)


def test_build_task_refuses_restricted_rows_outside_restricted_root(tmp_path):
    env = {**os.environ, "KEV_RESTRICTED_ROOT": str(tmp_path / "restricted")}
    args = [sys.executable, str(REPO / "scripts/data/build_task.py"), "--task", "p2", "--src",
            str(REPO / "tests/fixtures/physionet"), "--seed", "1", "--skip-bad"]
    r = subprocess.run(args + ["--out", str(tmp_path / "shared" / "p2")], capture_output=True, text=True, cwd=REPO, env=env)
    assert r.returncode == 4 and "restricted" in r.stderr
    r = subprocess.run(args + ["--out", str(tmp_path / "restricted" / "splits" / "p2")], capture_output=True, text=True, cwd=REPO, env=env)
    assert r.returncode == 0, r.stderr


def test_build_task_passes_extracted_at_to_i1(tmp_path):
    env = {**os.environ, "KEV_RESTRICTED_ROOT": str(tmp_path / "restricted")}
    args = [sys.executable, str(REPO / "scripts/data/build_task.py"), "--task", "i1", "--src",
            str(REPO / "tests/fixtures/internal/raw"), "--seed", "1", "--extracted-at", "2026-05-16T00:00:00+02:00",
            "--out", str(tmp_path / "restricted" / "splits" / "i1")]
    r = subprocess.run(args, capture_output=True, text=True, cwd=REPO, env=env)
    assert r.returncode == 0, r.stderr
    b = json.loads((tmp_path / "restricted" / "splits" / "i1" / "build.json").read_text())
    assert b["consort"]["dropped_open"] > 1 and b["extracted_at"] == "2026-05-16T00:00:00+02:00"
