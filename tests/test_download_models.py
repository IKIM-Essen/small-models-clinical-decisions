"""Pure-logic tests for scripts/data/download_models.py (no network: the Hub API and
snapshot_download are both injected as fakes)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "data" / "download_models.py"


@pytest.fixture(scope="module")
def dm():
    spec = importlib.util.spec_from_file_location("download_models", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["download_models"] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeApi:
    def __init__(self, sha_by_repo: dict[str, str]):
        self._sha = sha_by_repo
        self.calls: list[str] = []

    def model_info(self, repo_id: str):
        self.calls.append(repo_id)
        return type("Info", (), {"sha": self._sha[repo_id]})()


def test_pins_revision_on_first_resolution(dm):
    api = FakeApi({"Cloudflare/clef-flash": "aaaa111"})
    rev = dm.resolve_revision(api, "Cloudflare/clef-flash", pins={})
    assert rev == "aaaa111"
    assert api.calls == ["Cloudflare/clef-flash"]


def test_already_pinned_revision_is_reused_without_a_hub_call(dm):
    api = FakeApi({"Cloudflare/clef-flash": "bbbb222"})  # Hub HEAD moved on, pin must not follow it
    pins = {"Cloudflare/clef-flash": {"revision": "aaaa111"}}
    rev = dm.resolve_revision(api, "Cloudflare/clef-flash", pins=pins)
    assert rev == "aaaa111"
    assert api.calls == []  # no Hub call needed once pinned


def test_download_one_calls_snapshot_download_with_the_pinned_revision(dm, tmp_path):
    seen = {}

    def fake_snapshot_download(repo_id, revision, cache_dir, **kw):
        seen["args"] = (repo_id, revision, cache_dir)
        p = Path(cache_dir) / "fake-snapshot"
        p.mkdir(parents=True)
        (p / "weights.bin").write_bytes(b"x" * 100)
        return str(p)

    local = dm.download_one("Cloudflare/clef-flash", "aaaa111", tmp_path, snapshot_fn=fake_snapshot_download)
    assert seen["args"] == ("Cloudflare/clef-flash", "aaaa111", str(tmp_path))
    assert dm.dir_size_bytes(Path(local)) == 100


def test_dir_size_bytes_ignores_pycache(dm, tmp_path):
    # Importing a model's custom code (e.g. a smoke check) writes a __pycache__/*.pyc next to it;
    # that is incidental to running the code, not part of the downloaded asset, and must not make
    # --verify report a SIZE mismatch against the size recorded at download time.
    (tmp_path / "weights.bin").write_bytes(b"x" * 100)
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    (pycache / "joint_schema_model.cpython-313.pyc").write_bytes(b"y" * 32681)
    assert dm.dir_size_bytes(tmp_path) == 100


def test_main_only_downloads_requested_models_and_writes_pins(dm, tmp_path, monkeypatch):
    pins_path = tmp_path / "pins.json"
    pins_path.write_text(json.dumps({"kev_commit": "x"}))
    dest = tmp_path / "hf_home"

    def fake_snapshot_download(repo_id, revision, cache_dir, **kw):
        p = Path(cache_dir) / repo_id.replace("/", "--")
        p.mkdir(parents=True)
        (p / "f.bin").write_bytes(b"y" * 42)
        return str(p)

    monkeypatch.setattr(dm, "HfApi", lambda: FakeApi({"Cloudflare/clef-flash": "cccc333"}))
    monkeypatch.setattr(dm, "snapshot_download", fake_snapshot_download)

    rc = dm.main(["--only", "clef-flash", "--dest", str(dest), "--pins", str(pins_path)])
    assert rc == 0

    pins = json.loads(pins_path.read_text())
    assert pins["kev_commit"] == "x"  # untouched
    entry = pins["models"]["Cloudflare/clef-flash"]
    assert entry["revision"] == "cccc333"
    assert entry["local_size_bytes"] == 42
    assert "resolved_at" in entry


def test_verify_reports_missing_without_downloading(dm, tmp_path, monkeypatch):
    pins_path = tmp_path / "pins.json"
    pins_path.write_text(json.dumps({"models": {"Cloudflare/clef-flash": {
        "revision": "dddd444", "local_size_bytes": 999, "local_path": str(tmp_path / "nope")}}}))

    def boom(*a, **kw):
        raise AssertionError("--verify must not download")

    monkeypatch.setattr(dm, "snapshot_download", boom)
    rc = dm.main(["--verify", "--pins", str(pins_path)])
    assert rc == 1
