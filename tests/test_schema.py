"""Kevmed.schema: labelled Kev requests, validated against Kev's own request model."""
import json

import pytest

from kevmed import schema


def test_noul_choice_score_carry_labels():
    assert schema.noul("Is it malignant?", True)["label"] is True
    c = schema.choice("Which?", {"yes": "Yes", "no": None}, "no")
    assert c["type"] == "choice" and c["label"] == "no"
    s = schema.score("Acuity", ["low", "high"], 1)
    assert s["type"] == "score" and s["label"] == 1


def test_record_validates_with_kev_request_model():
    r = schema.record("x1", {"question": "q", "abstract": "a"},
                      {"ans": schema.choice("?", {"yes": None, "no": None, "maybe": None}, "maybe")},
                      {"src": "pubmedqa"})
    assert r["_meta"] == {"id": "x1", "src": "pubmedqa"}
    assert r["questions"]["ans"]["label"] == "maybe"


def test_record_rejects_label_outside_options():
    with pytest.raises(ValueError):
        schema.record("x2", "s", {"a": schema.choice("?", {"yes": None}, "no")})


def test_record_rejects_score_label_out_of_range():
    with pytest.raises(ValueError):
        schema.record("x3", "s", {"a": schema.score("?", ["low", "high"], 2)})


def test_record_rejects_empty_state():
    with pytest.raises(ValueError):
        schema.record("x4", "", {"a": schema.noul("?", False)})


def test_jsonl_roundtrip_and_summary(tmp_path):
    recs = [schema.record(f"r{i}", "s", {"a": schema.noul("?", i % 2 == 0)}) for i in range(4)]
    n = schema.write_jsonl(tmp_path / "x.jsonl", recs)
    assert n == 4 and schema.read_jsonl(tmp_path / "x.jsonl") == recs
    assert schema.summarize(recs) == {"n": 4, "questions": {"a": {"type": "noul", "labels": {"True": 2, "False": 2}}}}
    assert (tmp_path / "x.jsonl").read_text().count("\n") == 4


def test_record_refuses_meta_id_collision():
    with pytest.raises(ValueError):
        schema.record("real", "s", {"a": schema.noul("?", True)}, {"id": "other"})
