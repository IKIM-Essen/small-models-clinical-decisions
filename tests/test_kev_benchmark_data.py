"""Scripts/repro/kev_benchmark_data.py: strip `_meta` keys that collide with kev's own reserved
row-pairing fields before a record reaches kev.benchmark.

P2's builder (src/kevmed/tasks/p2_mednli.py) tags every record with
`_meta.pair_id` = MedNLI's own `pairID` (a provenance field, duplicate of `_meta.group`, used by
kevmed.splits for leakage-safe grouping). kev.data.load_records merges a record's own `_meta` into
the one it builds, and kev/benchmark.py's row builder reads `meta.get("pair_id")`/`meta.get("sibling")`
to reconstruct contrastive a/b sibling pairs (kev's own synthetic policy data, kev/contrastive.py)
for the `paired_flip` metric. Our `pair_id` is not a kev contrastive pair (no `sibling` field), so
`paired_flip` finds a `pair_id` group whose siblings never complete an {"a","b"} set and raises
"incomplete contrastive pair", crashing the whole report (rows/predictions still score fine; only
the aggregate summary crashes). Frozen splits are not touched (Section 6.a/6.c immutability) --
this strips the colliding key from a copy at evaluation time only.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "repro" / "kev_benchmark_data.py"


@pytest.fixture(scope="module")
def kbd():
    spec = importlib.util.spec_from_file_location("kev_benchmark_data", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kev_benchmark_data"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_strip_meta_keys_removes_only_named_keys(kbd):
    record = {"_meta": {"id": "x", "pair_id": "p1", "group": "p1", "src": "mednli"},
             "state": "...", "questions": {}}
    out = kbd.strip_meta_keys(record, {"pair_id"})
    assert "pair_id" not in out["_meta"]
    assert out["_meta"]["group"] == "p1" and out["_meta"]["id"] == "x"
    assert "pair_id" in record["_meta"]  # original untouched


def test_strip_meta_keys_is_a_noop_when_key_absent(kbd):
    record = {"_meta": {"id": "x"}, "state": "...", "questions": {}}
    out = kbd.strip_meta_keys(record, {"pair_id"})
    assert out == record


def test_filter_jsonl_strips_every_record_and_counts(kbd, tmp_path):
    src = tmp_path / "dev.jsonl"
    src.write_text(
        '{"_meta": {"id": "a", "pair_id": "1"}, "state": "s", "questions": {}}\n'
        '{"_meta": {"id": "b", "pair_id": "2"}, "state": "s", "questions": {}}\n'
    )
    dst = tmp_path / "dev.sanitized.jsonl"
    n = kbd.filter_jsonl(src, dst, {"pair_id"})
    assert n == 2
    lines = [json.loads(l) for l in dst.read_text().splitlines()]
    assert all("pair_id" not in r["_meta"] for r in lines)
    assert [r["_meta"]["id"] for r in lines] == ["a", "b"]
