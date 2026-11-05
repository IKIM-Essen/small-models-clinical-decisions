"""Labelled Kev requests: {"_meta": {...}, "state": ..., "questions": {id: {..., "label"}}}.

The shape is Kev's own (kev/data.py "labelled request"): choice label = option key, noul label = bool,
score label = level index. `record` validates the label-stripped request with kev.api.SystemOneRequest,
so a frozen split is byte-compatible with kev.train --data and kev.benchmark --data.
Aggregates only: nothing here prints rows.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from kev.api import SystemOneRequest


def noul(instructions: str, label: bool, criteria: dict | None = None) -> dict:
    q = {"type": "noul", "instructions": instructions, "label": bool(label)}
    if criteria:
        q["criteria"] = criteria
    return q


def choice(instructions: str, criteria: dict[str, str | None], label: str) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria), "label": label}


def score(instructions: str, criteria: list[str], label: int) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": list(criteria), "label": int(label)}


def _check_label(qid: str, q: dict) -> None:
    t, label = q["type"], q.get("label")
    if t == "noul" and not isinstance(label, bool):
        raise ValueError(f"question {qid}: noul label must be bool")
    if t == "choice" and label not in q["criteria"]:
        raise ValueError(f"question {qid}: choice label is not one of the options")
    if t == "score" and not (isinstance(label, int) and 0 <= label < len(q["criteria"])):
        raise ValueError(f"question {qid}: score label outside 0..{len(q['criteria']) - 1}")


def record(rid: str, state, questions: dict[str, dict], meta: dict | None = None) -> dict:
    if state in ("", None, {}, []):
        raise ValueError(f"record {rid}: empty state")
    if meta and "id" in meta and meta["id"] != rid:
        raise ValueError(f"record {rid}: meta['id'] collides with the record id")
    for qid, q in questions.items():
        _check_label(qid, q)
    stripped = {qid: {k: v for k, v in q.items() if k != "label"} for qid, q in questions.items()}
    try:
        SystemOneRequest(state=state, questions=stripped)
    except Exception as e:  # pydantic ValidationError; the message names fields, never row text
        raise ValueError(f"record {rid}: invalid request shape ({type(e).__name__})") from None
    return {"_meta": {"id": rid, **(meta or {})}, "state": state, "questions": questions}


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(records: list[dict]) -> dict:
    qs: dict[str, dict] = {}
    for r in records:
        for qid, q in r["questions"].items():
            entry = qs.setdefault(qid, {"type": q["type"], "labels": Counter()})
            entry["labels"][str(q["label"])] += 1
    return {"n": len(records), "questions": {qid: {"type": e["type"], "labels": dict(e["labels"])} for qid, e in qs.items()}}
