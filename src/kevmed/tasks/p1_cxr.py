"""P1 MIMIC-CXR v2.1.0 report text with CheXpert labels (mimic-cxr-2.0.0-chexpert.csv.gz) as 14 Kev `noul`
questions per report. Labels come from a rule-based labeler run on the same text
(ceiling stated in Methods, D8). 1.0 -> true, 0.0 and blank -> false, -1.0 -> question masked.
Official split file: train -> train, validate -> dev, test -> test. Never prints rows."""
from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

from .. import physionet, schema, splits

TASK, KEV_TYPE = "p1", "noul"
FINDINGS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
            "Lung Opacity", "Pleural Effusion", "Pneumonia", "Pneumothorax", "Pleural Other", "Support Devices", "No Finding"]
QID = {f: f.lower().replace(" ", "_") for f in FINDINGS}
SPLIT_MAP = {"train": "train", "validate": "dev", "test": "test"}
_WS = re.compile(r"[ \t]+")
_NL = re.compile(r"\n{3,}")


def report_path(subject_id: str, study_id: str) -> Path:
    return Path(physionet.LAYOUT["mimic_cxr_reports"]) / f"p{subject_id[:2]}" / f"p{subject_id}" / f"s{study_id}.txt"


def _instruction(finding: str) -> str:
    return "Does the report state no abnormal finding?" if finding == "No Finding" else f"Does the report state that {finding.lower()} is present?"


def _code(cell: str) -> float | None:
    """CheXpert cell -> 1.0 / 0.0 / -1.0; blank (not mentioned) -> 0.0; anything else -> None."""
    if cell == "":
        return 0.0
    try:
        x = float(cell)
    except ValueError:
        return None
    return x if x in (1.0, 0.0, -1.0) else None


def _clean(text: str) -> str:
    text = _WS.sub(" ", text.replace("\r", ""))
    return _NL.sub("\n\n", "\n".join(line.strip() for line in text.splitlines())).strip()


def build(src: Path, seed: int, skip_bad: bool = False) -> tuple[dict[str, list[dict]], dict]:
    src = Path(src)
    labels = {r["study_id"]: r for r in physionet.read_csv_gz(src / physionet.LAYOUT["mimic_cxr_chexpert"])}
    split_of, subject_of = {}, {}
    for r in physionet.read_csv_gz(src / physionet.LAYOUT["mimic_cxr_split"]):
        split_of[r["study_id"]] = SPLIT_MAP[r["split"]]
        subject_of[r["study_id"]] = r["subject_id"]
    consort = OrderedDict(studies_in_split_file=len(split_of), studies_with_report=0, dropped_missing_report=0,
                          dropped_no_chexpert_row=0, dropped_all_uncertain=0, dropped_empty_report=0, unexpected_chexpert_values=0)
    by = {"train": [], "dev": [], "test": []}
    for study, split in sorted(split_of.items()):
        subject = subject_of[study]
        path = src / report_path(subject, study)
        if not path.is_file():
            consort["dropped_missing_report"] += 1
            continue
        consort["studies_with_report"] += 1
        row = labels.get(study)
        if row is None:
            consort["dropped_no_chexpert_row"] += 1
            continue
        questions = {}
        for f in FINDINGS:
            v = (row.get(f) or "").strip()
            code = _code(v)
            if code is None:
                if not skip_bad:
                    raise ValueError(f"study {study}: unexpected CheXpert value in column {f}")
                consort["unexpected_chexpert_values"] += 1
                code = 0.0
            if code == -1.0:
                continue
            questions[QID[f]] = schema.noul(_instruction(f), code == 1.0)
        if not questions:
            consort["dropped_all_uncertain"] += 1
            continue
        text = _clean(path.read_text(encoding="utf-8", errors="replace"))
        if not text:
            consort["dropped_empty_report"] += 1
            continue
        by[split].append(schema.record(f"cxr-{study}", text, questions,
                                       {"study_id": study, "subject_id": subject, "group": subject, "src": "mimic-cxr"}))
    consort["subjects_overlapping_splits"] = splits.overlap(by, "subject_id")
    for k in by:
        consort[f"n_{k}"] = len(by[k])
    return by, consort
