"""Shared PhysioNet plumbing: the file layout under data/public/physionet/,
gzipped CSV streaming, the patient-level partition P3 and S1 share, and MIMIC-IV age arithmetic.
Never prints rows."""
from __future__ import annotations

import csv
import gzip
import random
from pathlib import Path
from typing import Iterable, Iterator

LAYOUT = {
    "mimic_cxr_reports": "mimic-cxr-2.1.0/files",
    "mimic_cxr_chexpert": "mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz",
    "mimic_cxr_split": "mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-split.csv.gz",
    "mednli": "mednli-1.0.0",
    "ed_edstays": "mimic-iv-ed-2.2/ed/edstays.csv.gz",
    "ed_triage": "mimic-iv-ed-2.2/ed/triage.csv.gz",
    "iv_patients": "mimic-iv-3.1/hosp/patients.csv.gz",
    "iii_noteevents": "mimic-iii-1.4/NOTEEVENTS.csv.gz",
    "phenotypes": "phenotype-annotations-mimic-1.20.03/ACTdb102003.csv",
}
csv.field_size_limit(2**31 - 1)
DEFAULT_FRACTIONS = {"train": 0.7, "dev": 0.1, "test": 0.2}
MIMIC_AGE_CAP = 91


def read_csv_gz(path: Path) -> Iterator[dict]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def patient_partition(subject_ids: Iterable[str], seed: int, fractions: dict[str, float] = DEFAULT_FRACTIONS) -> dict[str, str]:
    ids = sorted({str(s) for s in subject_ids})
    rng = random.Random(seed)
    rng.shuffle(ids)
    out, start, acc = {}, 0, 0.0
    names = list(fractions)
    for i, name in enumerate(names):
        acc += fractions[name]
        end = len(ids) if i == len(names) - 1 else round(acc * len(ids))
        for s in ids[start:end]:
            out[s] = name
        start = end
    return out


def age_at(anchor_age: int, anchor_year: int, at_year: int) -> int:
    return max(0, min(MIMIC_AGE_CAP, int(anchor_age) + (int(at_year) - int(anchor_year))))
