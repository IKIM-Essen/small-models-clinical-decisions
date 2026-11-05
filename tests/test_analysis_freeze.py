"""Scripts/analysis/freeze.py and scripts/analysis/testpass_job.py: freeze each run's configuration before its one
test pass, and render the test Job only for frozen runs. Synthetic run directories only."""
from __future__ import annotations

import base64
import importlib.util
import json
import re
from pathlib import Path

import pytest

ANALYSIS = Path(__file__).resolve().parents[1] / "scripts" / "analysis"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ANALYSIS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fz, tj = load("freeze"), load("testpass_job")


def fake_repo(tmp_path, runs=("kev4b-p2-3k-s0",), temperature=1.7):
    for name in runs:
        d = tmp_path / "runs" / name
        d.mkdir(parents=True)
        (d / "training_config.json").write_text("{}")
        dev = tmp_path / "runs" / f"{name}-dev"
        dev.mkdir()
        (dev / "report.json").write_text(json.dumps({"coverage": {"evaluated_records": 10}}))
        (dev / "calibration.json").write_text(json.dumps({"workload_temperature": temperature}))
    for path in fz.TEST_DATA.values():
        f = tmp_path / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text('{"synthetic": 1}\n')
    return tmp_path


def test_entry_records_temperature_test_file_hash_and_scorer(tmp_path):
    root = fake_repo(tmp_path)
    e = fz.entry(root, "kev4b-3k", "p2", 0, commit="abc123")
    assert e["run"] == "runs/kev4b-p2-3k-s0" and e["dev_temperature"] == 1.7
    assert e["test_data"] == "data/derived/p2-test-nopairid.jsonl" and len(e["test_data_sha256"]) == 64
    assert e["max_state"] == 1024 and e["scorer"] == "kev" and e["test_out"] == "runs/kev4b-p2-3k-s0-test"
    assert e["commit"] == "abc123"


def test_i2_3k_kev_row_is_its_full_data_run(tmp_path):
    root = fake_repo(tmp_path, runs=("kev4b-i2-full-s1",))
    e = fz.entry(root, "kev4b-3k", "i2", 1, commit="x")
    assert e["run"] == "runs/kev4b-i2-full-s1" and e["test_data"].endswith("i2-sf7400/test.novariant.jsonl")
    assert e["max_state"] == 7552


def test_entry_refuses_a_run_without_dev_calibration(tmp_path):
    root = fake_repo(tmp_path)
    (root / "runs" / "kev4b-p2-3k-s0-dev" / "calibration.json").unlink()
    with pytest.raises(fz.NotReady):
        fz.entry(root, "kev4b-3k", "p2", 0, commit="x")


def test_zero_shot_and_qwen_rows(tmp_path):
    root = fake_repo(tmp_path, runs=("kev4b-p1-zeroshot", "qwen4b-p1-zeroshot-k3", "sft4b-p1-3k-s2"))
    kz = fz.entry(root, "kev4b-zeroshot", "p1", 0, commit="x")
    assert kz["scorer"] == "kev" and kz["score_run"].endswith("models--jaredpalmer--kev-4b/snapshots/6cfce5c2fa4b4bd64026336ab649c5ca78857d52")
    qz = fz.entry(root, "qwen4b-zeroshot-k3", "p1", 0, commit="x")
    assert qz["scorer"] == "qwen" and qz["score_run"] == "base" and qz["shots"] == 3
    sft = fz.entry(root, "sft4b-3k", "p1", 2, commit="x")
    assert sft["scorer"] == "qwen" and sft["score_run"] == "runs/sft4b-p1-3k-s2" and sft["shots"] == 0


def test_freeze_adds_new_entries_and_never_changes_frozen_ones(tmp_path):
    root = fake_repo(tmp_path)
    out = root / "configs" / "frozen.json"
    fz.freeze(root, out, [("kev4b-3k", "p2", 0)], commit="c1")
    first = json.loads(out.read_text())
    fz.freeze(root, out, [("kev4b-3k", "p2", 0)], commit="c2")          # same entry again: kept as frozen first
    assert json.loads(out.read_text()) == first
    (root / "runs" / "kev4b-p2-3k-s0-dev" / "calibration.json").write_text(json.dumps({"workload_temperature": 9.0}))
    with pytest.raises(fz.FrozenChanged):
        fz.freeze(root, out, [("kev4b-3k", "p2", 0)], commit="c3")


def test_test_job_renders_the_scoring_command_for_frozen_entries_only(tmp_path):
    root = fake_repo(tmp_path)
    frozen = root / "configs" / "frozen.json"
    fz.freeze(root, frozen, [("kev4b-3k", "p2", 0)], commit="c1")
    template = (Path(__file__).resolve().parents[1] / "k8s" / "runner-job-gpu.yaml").read_text()
    y = tj.render(json.loads(frozen.read_text()), "kev4b-3k/p2/s0", template)
    cmd = base64.b64decode(re.search(r'name: RUNNER_CMD_B64, value: "([^"]+)"', y).group(1)).decode()
    assert "scripts/analysis/bench_ctx.py --max_state 1024 --run runs/kev4b-p2-3k-s0 --data data/derived/p2-test-nopairid.jsonl" in cmd
    assert "--out runs/kev4b-p2-3k-s0-test" in cmd and "kev.train" not in cmd and "calibrate" not in cmd
    assert "sha256sum -c" in cmd     # the Job re-checks the test file hash before scoring
    assert "name: kev-gpu-test-kev4b-3k-p2-s0" in y
    with pytest.raises(KeyError):
        tj.render(json.loads(frozen.read_text()), "sft4b-3k/p2/s0", template)


def test_qwen_test_job_uses_the_qwen_scorer(tmp_path):
    root = fake_repo(tmp_path, runs=("qwen4b-i1-zeroshot-k0",))
    frozen = root / "configs" / "frozen.json"
    fz.freeze(root, frozen, [("qwen4b-zeroshot-k0", "i1", 0)], commit="c1")
    template = (Path(__file__).resolve().parents[1] / "k8s" / "runner-job-gpu.yaml").read_text()
    y = tj.render(json.loads(frozen.read_text()), "qwen4b-zeroshot-k0/i1/s0", template)
    cmd = base64.b64decode(re.search(r'name: RUNNER_CMD_B64, value: "([^"]+)"', y).group(1)).decode()
    assert "scripts/qwen/bench_qwen.py --max_state 2048 --task i1 --shots 0 --run base --data splits/i1/test.jsonl" in cmd


def test_two_entries_never_share_a_test_output(tmp_path):
    root = fake_repo(tmp_path, runs=("kev4b-i2-full-s0",))
    out = root / "configs" / "frozen.json"
    fz.freeze(root, out, [("kev4b-3k", "i2", 0)], commit="c")
    with pytest.raises(fz.DuplicateTestOutput):
        fz.freeze(root, out, [("kev4b-full", "i2", 0)], commit="c")   # the same run: I2's full data is its 3k data
