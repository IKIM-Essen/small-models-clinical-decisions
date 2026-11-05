"""Label check as Excel workbooks for reviewers without pod access.

  uv run scripts/review/review_xlsx.py export --task s3 --n 10          # -> splits/review/xlsx/s3_review.xlsx
  uv run scripts/review/review_xlsx.py import --file s3_review.xlsx --reviewer gijs
  uv run scripts/review/label_check.py score                              # counts, same table as the terminal tool

One line per question: record text, question, the label, and a `verdict` dropdown (correct / incorrect / unsure) plus a
comment. The sample is the same as `label_check.py` (same seed, same rows). Where a workbook may travel:
- open tasks (s3, s4, s5): anywhere;
- internal tasks (i1 to i4): hospital infrastructure only (hospital PC, VPN, network drive), never private devices or
  external mail; the export refuses a target outside $KEV_RESTRICTED_ROOT, moving it onward is the reviewer's step;
- PhysioNet tasks (p1, p2, p3, s1, s2): no export, except `--credentialed <name>` for a reviewer who holds the
  PhysioNet credential and DUA themselves: written inside $KEV_RESTRICTED_ROOT, file marked for that person only,
  never passed on."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kevmed import schema, splits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_check import sample  # noqa: E402  (same rows as the terminal tool)

OPEN = {"s3", "s4", "s5"}
PHYSIONET = {"p1", "p2", "p3", "s1", "s2"}
VERDICTS = ("correct", "incorrect", "unsure")
HEADER = ["item", "record_id", "text", "question_id", "question", "label", "verdict", "comment"]
INSTRUCTIONS = [
    "Label check: is the label right for this text?",
    "Every line is one question about one record (records with several questions span several lines).",
    "Column 'verdict': choose correct, incorrect or unsure from the dropdown. 'comment' is optional.",
    "You judge whether the given label matches the text; you do not relabel.",
    "A record counts when all its lines have correct or incorrect; unsure or empty lines leave it out.",
    "Internal tasks (i1 to i4): keep this file on hospital infrastructure only, never on private devices or in external mail.",
]


CELL_MAX = 32767   # Excel's limit per cell


def _state_text(state) -> str:
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)
    if len(text) > CELL_MAX:
        cut = CELL_MAX - 60
        text = text[:cut] + f"\n... [{len(text) - cut} more characters, cut at the Excel cell limit]"
    return text


def _label_text(q: dict) -> str:
    label = q["label"]
    if q["type"] == "score" and isinstance(label, int) and "criteria" in q:
        return f"{label} ({q['criteria'][label]})"
    if q["type"] == "choice" and isinstance(q.get("criteria"), dict) and q["criteria"].get(label):
        return f"{label} ({q['criteria'][label]})"
    return str(label)


def export(task: str, split_path: Path, out_dir: Path, n: int = 10, seed: int = 20260925, suffix: str = "",
           credentialed: str | None = None) -> Path:
    import openpyxl
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation
    if task in PHYSIONET and not credentialed:
        raise PermissionError(f"{task} is PhysioNet data: the DUA forbids passing rows on, no workbook export "
                              "(--credentialed <name> only for a reviewer who holds the PhysioNet DUA)")
    if task not in OPEN:
        splits.require_restricted(Path(out_dir), f"review workbook for {task}")
    records = sample(schema.read_jsonl(split_path), n, seed)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "review"
    ws.append(HEADER)
    for c in ws[1]:
        c.font = Font(bold=True)
    for i, r in enumerate(records, 1):
        text = _state_text(r["state"])
        for qid, q in r["questions"].items():
            ws.append([i, r["_meta"]["id"], text, qid, q["instructions"], _label_text(q), None, None])
    last = ws.max_row
    dv = DataValidation(type="list", formula1='"' + ",".join(VERDICTS) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"G2:G{max(last, 2)}")
    for col, width in zip("ABCDEFGH", (6, 22, 90, 18, 45, 22, 12, 35)):
        ws.column_dimensions[col].width = width
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    info = wb.create_sheet("instructions")
    extra = [f"PhysioNet data: for {credentialed} only (holds the PhysioNet credential and DUA); never pass this file on, "
             "delete it after the review."] if task in PHYSIONET else []
    for line in INSTRUCTIONS + extra + [f"task: {task}", f"records: {len(records)}", f"seed: {seed}"]:
        info.append([line])
    info.column_dimensions["A"].width = 120
    tag = [suffix] if suffix else []
    if task in PHYSIONET:
        tag.append(f"PHYSIONET-{credentialed}-only")
    out = Path(out_dir) / ("_".join([f"{task}_review", *tag]) + ".xlsx")
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return out


def import_answers(xlsx: Path, reviewer: str, review_dir: Path) -> int:
    """Append one answer per fully judged record to <review_dir>/<task>/<reviewer>.jsonl (same format as the
    terminal tool, so `label_check.py score` counts both). Returns the number of records written."""
    import openpyxl
    wb = openpyxl.load_workbook(xlsx, read_only=True)
    task = next(str(r[0]).split(": ", 1)[1] for r in wb["instructions"].iter_rows(values_only=True)
                if r[0] and str(r[0]).startswith("task: "))
    if task not in OPEN:
        splits.require_restricted(Path(review_dir), f"review answers for {task}")
    by: dict[str, list[tuple]] = {}
    for row in wb["review"].iter_rows(min_row=2, values_only=True):
        if row[1] is None:
            continue
        by.setdefault(str(row[1]), []).append(row)
    out = Path(review_dir) / task / f"{reviewer}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    n = 0
    with out.open("a", encoding="utf-8") as f:
        for rid, lines in by.items():
            verdicts = [(str(l[6]).strip().lower() if l[6] else "") for l in lines]
            if rid in done or any(v not in ("correct", "incorrect") for v in verdicts):
                continue
            wrong = [str(l[3]) for l, v in zip(lines, verdicts) if v == "incorrect"]
            note = " | ".join(str(l[7]) for l in lines if l[7])
            f.write(json.dumps({"id": rid, "agree": not wrong, "wrong": wrong, "note": note, "source": "xlsx"}, ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("--task", required=True)
    ex.add_argument("--n", type=int, default=10)
    ex.add_argument("--seed", type=int, default=20260925)
    ex.add_argument("--split", default="dev")
    ex.add_argument("--splits-dir", type=Path, default=Path("splits"))
    ex.add_argument("--out", type=Path, default=None, help="default splits/review/xlsx")
    ex.add_argument("--suffix", default="", help="file name suffix, e.g. no-doc")
    ex.add_argument("--credentialed", default=None, help="PhysioNet tasks only: name of the reviewer who holds the DUA")
    im = sub.add_parser("import")
    im.add_argument("--file", type=Path, required=True)
    im.add_argument("--reviewer", required=True)
    im.add_argument("--splits-dir", type=Path, default=Path("splits"))
    a = ap.parse_args(argv)
    if a.cmd == "export":
        try:
            p = export(a.task, a.splits_dir / a.task / f"{a.split}.jsonl", a.out or a.splits_dir / "review" / "xlsx", a.n, a.seed,
                       suffix=a.suffix, credentialed=a.credentialed)
        except PermissionError as e:
            print(e, file=sys.stderr)
            return 4
        print(f"wrote {p}")
        return 0
    print(f"imported {import_answers(a.file, a.reviewer, a.splits_dir / 'review')} records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
