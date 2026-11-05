"""Scripts/repro/run_manifest.py: every run gets a manifest next to metrics.json (Section 6.a)."""
import json
import subprocess
import sys

from conftest import REPO, load_script

REQUIRED = {"run_id", "git_commit", "git_dirty", "config", "split_hashes", "model",
            "deployment_tag", "seed", "start", "end", "host", "gpu", "python"}


def test_build_manifest_has_every_required_key():
    rm = load_script("scripts/repro/run_manifest.py")
    m = rm.build_manifest("r1", "jaredpalmer/kev-0.8b", {"suite": "x"}, ["abc"], 0, None, None, "mlx", REPO)
    assert REQUIRED <= set(m)
    assert m["git_commit"] and len(m["git_commit"]) == 40
    assert m["config"] == {"suite": "x"} and m["split_hashes"] == ["abc"]


def test_cli_writes_manifest(tmp_path):
    out = tmp_path / "runs" / "r2"
    r = subprocess.run([sys.executable, str(REPO / "scripts/repro/run_manifest.py"), "--run-id", "r2",
                        "--model", "m", "--config", '{"a": 1}', "--runs-dir", str(tmp_path / "runs")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = json.loads((out / "manifest.json").read_text())
    assert m["config"] == {"a": 1} and m["run_id"] == "r2"


def test_cli_rejects_bad_config_json(tmp_path):
    r = subprocess.run([sys.executable, str(REPO / "scripts/repro/run_manifest.py"), "--run-id", "r3",
                        "--model", "m", "--config", "{not json", "--runs-dir", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "JSON" in r.stderr


def test_cli_refuses_to_write_without_a_git_commit(tmp_path):
    """A manifest without a commit cannot trace a number to code (Section 6.a)."""
    r = subprocess.run([sys.executable, str(REPO / "scripts/repro/run_manifest.py"), "--run-id", "r4",
                        "--model", "m", "--root", str(tmp_path), "--runs-dir", str(tmp_path / "runs")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "git" in r.stderr
    assert not (tmp_path / "runs" / "r4" / "manifest.json").exists()


def test_root_defaults_to_the_scripts_own_repo(tmp_path):
    """Invoked from another directory, the manifest still records this repo's commit."""
    r = subprocess.run([sys.executable, str(REPO / "scripts/repro/run_manifest.py"), "--run-id", "r5",
                        "--model", "m", "--runs-dir", str(tmp_path / "runs")],
                       capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    m = json.loads((tmp_path / "runs" / "r5" / "manifest.json").read_text())
    assert len(m["git_commit"]) == 40
