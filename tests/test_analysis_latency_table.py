import importlib.util, json
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "latency_table.py"

@pytest.fixture(scope="module")
def lt():
    spec = importlib.util.spec_from_file_location("latency_table", SCRIPT)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def write(tmp, name, latency, remote=None, n=10, split="custom"):
    d = tmp / name; d.mkdir()
    (d / "report.json").write_text(json.dumps({"latency_ms": latency, "remote": remote, "split": split,
                                                 "coverage": {"evaluated_records": n}}))
    return d / "report.json"

def test_collect_reads_p50_p95_and_remote_flag(lt, tmp_path):
    a = write(tmp_path, "kev4b-p2-full-s0-dev", {"median": 59.2, "p95": 84.5})
    b = write(tmp_path, "frontier-kimi-k3-nothink-k0-p2-test", {"median": 9892.8, "p95": 18083.4},
              remote={"base_url": "x", "concurrency": 48})
    rows = {r["run"]: r for r in lt.collect([a, b])}
    assert rows["kev4b-p2-full-s0-dev"]["p50_ms"] == 59.2 and rows["kev4b-p2-full-s0-dev"]["remote"] is False
    assert rows["frontier-kimi-k3-nothink-k0-p2-test"]["remote"] is True
    assert rows["frontier-kimi-k3-nothink-k0-p2-test"]["concurrency"] == 48

def test_collect_skips_reports_without_latency(lt, tmp_path):
    a = write(tmp_path, "x", None)
    assert lt.collect([a]) == []
