import json

import openpyxl
import pytest

from conftest import load_script
from kevmed import schema

rx = load_script("scripts/review/review_xlsx.py")


def _write_split(path, n=12):
    recs = [schema.record(f"r{i}", f"SYNTHETIC state {i}", {"q": schema.noul("Is it?", i % 2 == 0),
                                                           "k": schema.choice("Which?", {"a": None, "b": None}, "a")}) for i in range(n)]
    schema.write_jsonl(path, recs)


def test_export_one_line_per_question_with_dropdown(tmp_path, monkeypatch):
    monkeypatch.setenv("KEV_RESTRICTED_ROOT", str(tmp_path))
    _write_split(tmp_path / "splits" / "s3" / "dev.jsonl")
    out = rx.export("s3", tmp_path / "splits" / "s3" / "dev.jsonl", tmp_path / "out", n=10, seed=1)
    wb = openpyxl.load_workbook(out)
    ws = wb["review"]
    header = [c.value for c in ws[1]]
    assert header[:7] == ["item", "record_id", "text", "question_id", "question", "label", "verdict"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    assert len(rows) == 20 and {r[1] for r in rows} == {r[1] for r in rows[:20]} and len({r[1] for r in rows}) == 10
    assert any(dv.type == "list" and "correct" in dv.formula1 for dv in ws.data_validations.dataValidation)
    assert "instructions" in wb.sheetnames


def test_restricted_tasks_export_only_inside_the_restricted_root(tmp_path, monkeypatch):
    monkeypatch.setenv("KEV_RESTRICTED_ROOT", str(tmp_path / "root"))
    _write_split(tmp_path / "root" / "splits" / "i3" / "dev.jsonl")
    with pytest.raises(PermissionError):
        rx.export("i3", tmp_path / "root" / "splits" / "i3" / "dev.jsonl", tmp_path / "outside", n=10, seed=1)
    for t in ("p1", "p2", "p3", "s1", "s2"):
        with pytest.raises(PermissionError):
            rx.export(t, tmp_path / "root" / "splits" / "i3" / "dev.jsonl", tmp_path / "root" / "x", n=10, seed=1)
    assert rx.export("i3", tmp_path / "root" / "splits" / "i3" / "dev.jsonl", tmp_path / "root" / "review_out", n=10, seed=1).exists()


def test_import_turns_verdicts_into_review_answers(tmp_path, monkeypatch):
    monkeypatch.setenv("KEV_RESTRICTED_ROOT", str(tmp_path))
    _write_split(tmp_path / "splits" / "s3" / "dev.jsonl")
    out = rx.export("s3", tmp_path / "splits" / "s3" / "dev.jsonl", tmp_path / "out", n=3, seed=1)
    wb = openpyxl.load_workbook(out); ws = wb["review"]
    verdicts = ["correct", "correct", "incorrect", "correct", "unsure", ""]
    for i, v in enumerate(verdicts, start=2):
        ws.cell(row=i, column=7, value=v or None)
    ws.cell(row=4, column=8, value="SYNTHETIC comment")
    wb.save(out)
    n = rx.import_answers(out, "gijs", tmp_path / "splits" / "review")
    rows = [json.loads(l) for l in (tmp_path / "splits" / "review" / "s3" / "gijs.jsonl").read_text().splitlines()]
    assert n == 2 and [r["agree"] for r in rows] == [True, False]
    assert rows[1]["wrong"] == ["q"] and rows[1]["note"] == "SYNTHETIC comment"
    assert all(r["source"] == "xlsx" for r in rows)


def test_physionet_export_only_for_a_named_credentialed_reviewer_inside_the_root(tmp_path, monkeypatch):
    monkeypatch.setenv("KEV_RESTRICTED_ROOT", str(tmp_path / "root"))
    split = tmp_path / "root" / "splits" / "p1" / "dev.jsonl"
    _write_split(split)
    with pytest.raises(PermissionError):
        rx.export("p1", split, tmp_path / "outside", n=10, seed=1, credentialed="reviewer_a")
    out = rx.export("p1", split, tmp_path / "root" / "x", n=10, seed=1, credentialed="reviewer_a")
    assert out.name == "p1_review_PHYSIONET-reviewer_a-only.xlsx"
    lines = [r[0] for r in openpyxl.load_workbook(out)["instructions"].iter_rows(values_only=True)]
    assert any("reviewer_a" in l and "PhysioNet" in l for l in lines)


def test_suffix_and_long_text_is_capped_at_the_excel_cell_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("KEV_RESTRICTED_ROOT", str(tmp_path))
    split = tmp_path / "splits" / "s3" / "dev.jsonl"
    schema.write_jsonl(split, [schema.record("r0", "x" * 40000, {"q": schema.noul("Is it?", True)})])
    out = rx.export("s3", split, tmp_path / "out", n=10, seed=1, suffix="no-doc")
    assert out.name == "s3_review_no-doc.xlsx"
    text = openpyxl.load_workbook(out)["review"]["C2"].value
    assert len(text) <= 32767 and "more characters" in text
