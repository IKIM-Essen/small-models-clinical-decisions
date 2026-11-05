import json

import pytest

from kevmed import schema, tokens


@pytest.fixture(scope="module")
def tok():
    return tokens.load_tokenizer()


def test_histogram_shape():
    h = tokens.histogram([10, 200, 400, 1100, 8000])
    assert h["n"] == 5 and h["min"] == 10 and h["max"] == 8000 and h["over_384"] == 3 and h["over_7552"] == 1
    assert sum(h["bins"].values()) == 5 and h["p50"] == 400


def test_state_text_matches_kev_render():
    assert tokens.state_text({"a": "x", "b": ["y"]}) == "a: x\nb:\n  - y"


def test_split_stats_reports_no_text(tmp_path, tok):
    recs = [schema.record(f"r{i}", "word " * (50 * (i + 1)), {"q": schema.noul("Is it?", True)}) for i in range(3)]
    schema.write_jsonl(tmp_path / "dev.jsonl", recs)
    s = tokens.split_stats(tmp_path / "dev.jsonl", tok)
    assert s["state"]["n"] == 3 and s["state"]["max"] > s["state"]["min"] and s["question_max"] > 0
    assert "word" not in json.dumps(s)


def test_histogram_of_empty_list_does_not_crash():
    h = tokens.histogram([])
    assert h["n"] == 0 and h["max"] is None and h["over_384"] == 0
