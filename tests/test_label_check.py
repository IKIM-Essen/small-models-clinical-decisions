import json

from conftest import REPO, load_script
from kevmed import schema

lc = load_script("scripts/review/label_check.py")


def _records(n=8):
    return [schema.record(f"r{i}", f"SYNTHETIC state {i}", {"q": schema.noul("Is it?", i % 2 == 0), "k": schema.choice("Which?", {"a": None, "b": None}, "a")}, {"src": "fx"}) for i in range(n)]


def test_sample_is_deterministic_and_capped():
    recs = _records(8)
    a = lc.sample(recs, 5, seed=20260925)
    b = lc.sample(recs, 5, seed=20260925)
    assert [r["_meta"]["id"] for r in a] == [r["_meta"]["id"] for r in b] and len(a) == 5
    assert len(lc.sample(recs, 50, seed=1)) == 8


def test_review_records_answers_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setenv("KEV_RESTRICTED_ROOT", str(tmp_path))
    split = tmp_path / "splits" / "s3"
    schema.write_jsonl(split / "dev.jsonl", _records(4))
    answers = iter(["y", "n", "q", "", "q"])
    shown = []
    n = lc.run_review("s3", split / "dev.jsonl", tmp_path / "splits" / "review", "reviewer_a", 3, 20260925,
                      ask=lambda prompt: next(answers), show=shown.append)
    out = tmp_path / "splits" / "review" / "s3" / "reviewer_a.jsonl"
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert n == 2 and [r["agree"] for r in rows] == [True, False] and rows[1]["wrong"] == ["q"]
    assert any("SYNTHETIC state" in s for s in shown) and all("label" not in s.lower() or "True" in s or "a" in s for s in shown)
    answers = iter(["y"])
    n = lc.run_review("s3", split / "dev.jsonl", tmp_path / "splits" / "review", "reviewer_a", 3, 20260925,
                      ask=lambda prompt: next(answers), show=shown.append)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert n == 1 and len(rows) == 3 and len({r["id"] for r in rows}) == 3


def test_review_refuses_output_outside_restricted_root(tmp_path, monkeypatch):
    monkeypatch.setenv("KEV_RESTRICTED_ROOT", str(tmp_path / "restricted"))
    split = tmp_path / "restricted" / "splits" / "s3"
    schema.write_jsonl(split / "dev.jsonl", _records(2))
    try:
        lc.run_review("s3", split / "dev.jsonl", tmp_path / "elsewhere", "x", 2, 1, ask=lambda p: "y", show=lambda s: None)
        assert False
    except PermissionError:
        pass


def test_score_counts_only(tmp_path, capsys):
    d = tmp_path / "review"
    (d / "s3").mkdir(parents=True)
    (d / "s3" / "reviewer_a.jsonl").write_text('{"id": "r0", "agree": true, "wrong": []}\n{"id": "r1", "agree": false, "wrong": ["q"], "note": "SYNTHETIC note"}\n')
    (d / "s4").mkdir()
    (d / "s4" / "reviewer_b.jsonl").write_text('{"id": "r0", "agree": true, "wrong": []}\n')
    table = lc.score(d)
    assert table == {"s3": {"reviewer_a": {"reviewed": 2, "agree": 1, "wrong_by_question": {"q": 1}}}, "s4": {"reviewer_b": {"reviewed": 1, "agree": 1, "wrong_by_question": {}}}}
    lc.print_score(table)
    out = capsys.readouterr().out
    assert "s3" in out and "1 / 2" in out and "SYNTHETIC" not in out


def test_wrong_questions_accept_numbers_or_ids_and_reject_anything_else(tmp_path, monkeypatch):
    """2026-10-01: reviewers typed record ids and bare indices into the wrong-questions prompt, and `score` printed
    them (MedNLI pair ids in the output). Questions are now numbered 1..k; numbers and question ids are accepted,
    anything else is asked again."""
    monkeypatch.setenv("KEV_RESTRICTED_ROOT", str(tmp_path))
    split = tmp_path / "splits" / "s5"
    schema.write_jsonl(split / "dev.jsonl", _records(1))
    answers = iter(["n", "r0", "7", "2", "", "q"])
    shown = []
    lc.run_review("s5", split / "dev.jsonl", tmp_path / "splits" / "review", "sujan", 1, 1, ask=lambda p: next(answers), show=shown.append)
    row = json.loads((tmp_path / "splits" / "review" / "s5" / "sujan.jsonl").read_text().splitlines()[0])
    assert row["wrong"] == ["k"]
    assert any("1. [q]" in s and "2. [k]" in s for s in shown)


def test_score_never_prints_entries_that_are_not_question_ids(tmp_path, capsys):
    d = tmp_path / "review"
    (d / "p2").mkdir(parents=True)
    (d / "p2" / "sujan.jsonl").write_text('{"id": "mednli-abc", "agree": false, "wrong": ["mednli-SYNTHETIC-1"]}\n'
                                          '{"id": "mednli-def", "agree": false, "wrong": ["relation"]}\n')
    table = lc.score(d, valid_questions={"p2": {"relation"}})
    assert table["p2"]["sujan"]["wrong_by_question"] == {"relation": 1, "(unspecified)": 1}
    lc.print_score(table)
    assert "SYNTHETIC" not in capsys.readouterr().out
