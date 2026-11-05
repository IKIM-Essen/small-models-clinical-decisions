"""Scripts/analysis/latency.py: per-decision latency for Kev from kev.benchmark
predictions and for the frontier tier from its call ledgers. Synthetic files only."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "latency.py"


@pytest.fixture(scope="module")
def lat():
    spec = importlib.util.spec_from_file_location("latency", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_kev_per_decision_divides_record_latency_by_its_questions(lat, tmp_path):
    f = tmp_path / "predictions.jsonl"
    f.write_text("\n".join(json.dumps({"id": f"r{i}", "rows": [{}] * q, "prediction": {"latency_ms": ms}})
                           for i, (q, ms) in enumerate([(1, 50.0), (2, 100.0), (14, 1400.0)])) + "\n")
    r = lat.kev_per_decision(f)
    assert r["per_decision_ms"] == [50.0, 50.0, 100.0] and r["p50_ms"] == pytest.approx(50.0) and r["n_records"] == 3


def test_frontier_uses_successful_calls_only(lat, tmp_path):
    f = tmp_path / "ledger.jsonl"
    f.write_text("\n".join(json.dumps({"status": s, "latency_s": v}) for s, v in
                           [("ok", 1.0), ("ok", 3.0), ("rate_limit", 99.0), ("ok", 2.0)]) + "\n")
    r = lat.frontier_per_call([f])
    assert r["n_calls"] == 3 and r["p50_ms"] == pytest.approx(2000.0)


def test_speedup_is_frontier_over_kev(lat):
    assert lat.speedup({"p50_ms": 2000.0}, {"p50_ms": 50.0}) == pytest.approx(40.0)


def test_pinned_job_runs_on_one_gpu_product_and_scores_the_snapshot(lat):
    template = (Path(__file__).resolve().parents[1] / "k8s" / "runner-job-gpu.yaml").read_text()
    y = lat.render_job(template)
    import base64, re
    assert "values: [NVIDIA-A100-SXM4-80GB]" in y and "NVIDIA-A100-80GB-PCIe" not in y
    cmd = base64.b64decode(re.search(r'name: RUNNER_CMD_B64, value: "([^"]+)"', y).group(1)).decode()
    assert "models--jaredpalmer--kev-4b/snapshots/6cfce5c2" in cmd and cmd.count("bench_ctx.py") == 6
    assert "data/derived/latency/p2-dev500.jsonl" in cmd and "--allow-test" not in cmd and "test.jsonl" not in cmd
