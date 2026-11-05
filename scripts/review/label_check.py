"""Human label check on the pod: 50 dev rows per task, one reviewer at a time, answers kept
as row-level files under $KEV_RESTRICTED_ROOT/splits/review/<task>/<reviewer>.jsonl (never in git);
`score` prints counts only.

  uv run scripts/review/label_check.py review --task p2 --reviewer reviewer_a        # interactive, resumable
  uv run scripts/review/label_check.py review --task i1 --reviewer reviewer_b --n 20
  uv run scripts/review/label_check.py score                                         # agreement table, counts only

Per record the state and every question with its label are shown; answer `y` (all labels right), `n` (then name the
wrong questions by number or id, comma-separated, empty = all wrong), `s` (skip for now) or `q` (quit; rerun to continue)."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from kevmed import schema, splits

TASKS = ["s3", "s4", "s5", "p1", "p2", "p3", "s1", "s2", "i1", "i2", "i2-nodiag", "i3", "i4"]
DEFAULT_N = 50


def sample(records: list[dict], n: int, seed: int) -> list[dict]:
    """Deterministic sample of `n` records (all of them when fewer), same for every reviewer."""
    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    return [records[i] for i in sorted(idx[:n])]


def _render(i: int, total: int, r: dict, max_chars: int) -> str:
    state = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False, indent=1)
    if len(state) > max_chars:
        state = state[:max_chars] + f"\n... [{len(state) - max_chars} more characters]"
    lines = [f"\n===== record {i}/{total}  id={r['_meta']['id']} =====", state, "----- questions -----"]
    for n, (qid, q) in enumerate(r["questions"].items(), 1):
        label = q["label"]
        if q["type"] == "score" and isinstance(label, int) and "criteria" in q:
            label = f"{label} ({q['criteria'][label]})"
        lines.append(f"{n}. [{qid}] {q['instructions']}\n    label: {label}")
    return "\n".join(lines)


def _done(out: Path) -> set[str]:
    if not out.exists():
        return set()
    return {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}


def run_review(task: str, split_path: Path, review_dir: Path, reviewer: str, n: int, seed: int, ask=input,
               show=print, max_chars: int = 12000) -> int:
    """Interactive loop; returns the number of records answered in this session."""
    splits.require_restricted(Path(review_dir), "label check answers")
    records = sample(schema.read_jsonl(split_path), n, seed)
    out = Path(review_dir) / task / f"{reviewer}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done(out)
    answered = 0
    with out.open("a", encoding="utf-8") as f:
        for i, r in enumerate(records, 1):
            rid = r["_meta"]["id"]
            if rid in done:
                continue
            show(_render(i, len(records), r, max_chars))
            while True:
                a = (ask("all labels correct? [y/n/s/q] ") or "").strip().lower()
                if a in ("y", "n", "s", "q"):
                    break
            if a == "q":
                break
            if a == "s":
                continue
            wrong: list[str] = []
            note = ""
            if a == "n":
                wrong = _ask_wrong(ask, list(r["questions"]))
                note = (ask("note (optional): ") or "").strip()
            f.write(json.dumps({"id": rid, "agree": a == "y", "wrong": wrong, "note": note}, ensure_ascii=False) + "\n")
            f.flush()
            answered += 1
    show(f"\n{answered} answered this session, {len(done) + answered} of {len(records)} in {out}")
    return answered


def _ask_wrong(ask, qids: list[str]) -> list[str]:
    """Question numbers (1..k, as shown) or question ids, comma-separated; empty = all. Anything else is asked again,
    so record ids or free text never end up in the answers (2026-10-01)."""
    while True:
        raw = (ask(f"which questions are wrong? numbers 1-{len(qids)} or ids, comma-separated (empty = all): ") or "").strip()
        if not raw:
            return list(qids)
        picked, ok = [], True
        for tok in (t.strip() for t in raw.split(",") if t.strip()):
            if tok.isdigit() and 1 <= int(tok) <= len(qids):
                picked.append(qids[int(tok) - 1])
            elif tok in qids:
                picked.append(tok)
            else:
                ok = False
        if ok and picked:
            return list(dict.fromkeys(picked))


def question_ids(splits_dir: Path, tasks) -> dict[str, set[str]]:
    """Question ids per task from the first record of its dev split (keys only, no row content)."""
    out = {}
    for t in tasks:
        f = Path(splits_dir) / t / "dev.jsonl"
        if f.exists():
            with f.open(encoding="utf-8") as fh:
                first = fh.readline()
            if first.strip():
                out[t] = set(json.loads(first)["questions"])
    return out


def score(review_dir: Path, valid_questions: dict[str, set[str]] | None = None) -> dict:
    table: dict = {}
    for task_dir in sorted(Path(review_dir).glob("*/")):
        for f in sorted(task_dir.glob("*.jsonl")):
            rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
            valid = (valid_questions or {}).get(task_dir.name)
            wrong = Counter((q if valid is None or q in valid else "(unspecified)") for r in rows for q in r.get("wrong", []))
            table.setdefault(task_dir.name, {})[f.stem] = {"reviewed": len(rows), "agree": sum(1 for r in rows if r["agree"]),
                                                          "wrong_by_question": dict(sorted(wrong.items()))}
    return table


def print_score(table: dict) -> None:
    print(f"{'task':<10} {'reviewer':<12} {'agree':>12}   wrong by question")
    for task, by in table.items():
        for reviewer, s in by.items():
            wq = ", ".join(f"{q} {n}" for q, n in s["wrong_by_question"].items()) or "-"
            print(f"{task:<10} {reviewer:<12} {s['agree']:>5} / {s['reviewed']:<4}   {wq}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rv = sub.add_parser("review")
    rv.add_argument("--task", choices=TASKS, required=True)
    rv.add_argument("--reviewer", required=True, help="your first name; one answer file per reviewer")
    rv.add_argument("--split", default="dev", help="dev (default) or test/train")
    rv.add_argument("--n", type=int, default=DEFAULT_N)
    rv.add_argument("--seed", type=int, default=20260925)
    rv.add_argument("--splits-dir", type=Path, default=Path("splits"))
    rv.add_argument("--max-chars", type=int, default=12000)
    sc = sub.add_parser("score")
    sc.add_argument("--splits-dir", type=Path, default=Path("splits"))
    a = ap.parse_args(argv)
    review_dir = a.splits_dir / "review"
    if a.cmd == "review":
        split_path = a.splits_dir / a.task / f"{a.split}.jsonl"
        if not split_path.exists():
            print(f"no split file {split_path} (run on the pod after the freeze)", file=sys.stderr)
            return 2
        run_review(a.task, split_path, review_dir, a.reviewer, a.n, a.seed, max_chars=a.max_chars)
        return 0
    tasks = [d.name for d in review_dir.glob('*/') if d.is_dir()]
    print_score(score(review_dir, question_ids(a.splits_dir, tasks)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
