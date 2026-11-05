"""I2 ICD-10-GM chapter of the DRG principal diagnosis from the internal discharge letter (Kev `choice`, 22 options),
German long-document task and the H4b test. Label: the single `drg` Condition of an IMP case
mapped by code range (kevmed.icd). Text: OCR-sidecar markdown of the latest final KDL AD010103 letter, ICD code strings
stripped; variant "no_diagnosis" removes the diagnosis section (reading-vs-deciding ablation). Years 2023 to 2026;
seeded chapter-stratified patient-level sample of at most 3k / 1k / 2k."""
from __future__ import annotations

import json
import random
from collections import OrderedDict
from pathlib import Path

from .. import context, icd, schema, splits

TASK, KEV_TYPE = "i2", "choice"
INSTRUCTIONS = "Which ICD-10-GM chapter does the principal diagnosis of this hospital stay belong to?"
DIAGNOSIS_HEADINGS = {"diagnosen", "diagnose", "hauptdiagnose", "nebendiagnosen", "diagnosen bei entlassung"}
DEFAULT_SAMPLE = {"train": 3000, "dev": 1000, "test": 2000}
MIN_CHARS = 200
ROMAN = {num: roman for num, roman, *_ in icd.CHAPTERS}


def split_of_year(year: int) -> str:
    return "train" if year <= 2024 else "dev" if year == 2025 else "test"


def remove_diagnosis_sections(text: str) -> str:
    secs = context._sections(text, DIAGNOSIS_HEADINGS)
    kept = [f"{h.capitalize()}\n{b}".strip() if h else b.strip() for h, b in secs if h not in DIAGNOSIS_HEADINGS]
    return "\n".join(k for k in kept if k)


def _stratified_sample(records: list[dict], n: int, seed: int) -> list[dict]:
    """Chapter-stratified, patient-grouped draw of exactly n records (top-up / trim with the same seed when the
    per-stratum rounding misses n)."""
    if len(records) <= n:
        return records
    groups = {r["_meta"]["id"]: r["_meta"]["group"] for r in records}
    labels = {r["_meta"]["id"]: r["questions"]["chapter"]["label"] for r in records}
    keep = n / len(records)
    chosen = splits.group_split(groups, labels, {"keep": keep, "drop": 1 - keep}, seed)
    kept = [r for r in records if chosen[r["_meta"]["id"]] == "keep"]
    rest = [r for r in records if chosen[r["_meta"]["id"]] != "keep"]
    rng = random.Random(seed)
    if len(kept) > n:
        rng.shuffle(kept)
        kept = kept[:n]
    elif len(kept) < n:
        rng.shuffle(rest)
        kept += rest[: n - len(kept)]
    ids = {r["_meta"]["id"] for r in kept}
    return [r for r in records if r["_meta"]["id"] in ids]


def latest_letters(rows: list[dict]) -> tuple[dict[str, dict], int]:
    """One letter per case, the latest `doc_date`; returns (case_id -> row, number of extra letters)."""
    latest: dict[str, dict] = {}
    extra = 0
    for r in rows:
        cur = latest.get(r["case_id"])
        if cur is None or r["doc_date"] > cur["doc_date"]:
            if cur is not None:
                extra += 1
            latest[r["case_id"]] = r
        else:
            extra += 1
    return latest, extra


def select_sample(rows: list[dict], seed: int, sample: dict[str, int] | None = None, oversample: float = 1.05) -> dict[str, list[str]]:
    """Doc ids to OCR: the chapter-stratified, patient-level sample per split drawn on the raw rows before OCR, with
    `oversample` headroom for letters that will drop after OCR (no text, too short)."""
    sample = sample or DEFAULT_SAMPLE
    latest, _ = latest_letters(rows)
    pools = {"train": [], "dev": [], "test": []}
    for r in latest.values():
        chapter = icd.icd10_chapter(r["drg_code"] or "")
        if chapter is None:
            continue
        pools[split_of_year(int(r["year"]))].append({"_meta": {"id": r["doc_id"], "group": r["patient_id"]},
                                                     "questions": {"chapter": {"label": ROMAN[chapter]}}})
    out = {}
    for i, (k, v) in enumerate(pools.items()):
        n = int(round(sample.get(k, len(v)) * oversample))
        out[k] = [r["_meta"]["id"] for r in _stratified_sample(v, n, seed + i)]
    return out


def build(src: Path, seed: int, skip_bad: bool = False, variant: str = "main", sample: dict[str, int] | None = None) -> tuple[dict[str, list[dict]], dict]:
    src = Path(src) / "i2"
    sample = sample or DEFAULT_SAMPLE
    rows = schema.read_jsonl(src / "rows.jsonl")
    latest, extra = latest_letters(rows)
    ledger = src / "ocr" / "ledger.jsonl"
    attempted = None
    if ledger.is_file():
        attempted = {json.loads(l)["doc_id"] for l in ledger.read_text(encoding="utf-8").splitlines() if l.strip()}
    consort = OrderedDict(letters=len(rows), cases=len(latest), extra_letters=extra, dropped_unmapped_code=0,
                          not_selected=0, dropped_no_ocr=0, dropped_short_text=0, codes_stripped=0)
    pools = {"train": [], "dev": [], "test": []}
    options = icd.chapter_options()
    for r in latest.values():
        chapter = icd.icd10_chapter(r["drg_code"] or "")
        if chapter is None:
            consort["dropped_unmapped_code"] += 1
            continue
        md = src / "ocr" / f"{r['doc_id']}.md"
        if not md.is_file():
            if attempted is not None and r["doc_id"] not in attempted:
                consort["not_selected"] += 1          # never sent to OCR (outside the pre-OCR sample), not a failure
            else:
                consort["dropped_no_ocr"] += 1
            continue
        text = md.read_text(encoding="utf-8")
        if variant == "no_diagnosis":
            text = remove_diagnosis_sections(text)
        text, n = icd.strip_icd_codes(text)
        consort["codes_stripped"] += n
        text = text.strip()
        if len(text) < MIN_CHARS:
            consort["dropped_short_text"] += 1
            continue
        year = int(r["year"])
        pools[split_of_year(year)].append(schema.record(f"letter-{r['doc_id']}", text, {"chapter": schema.choice(INSTRUCTIONS, options, ROMAN[chapter])},
            {"case_id": r["case_id"], "patient_id": r["patient_id"], "group": r["patient_id"], "doc_id": r["doc_id"], "year": year,
             "icd_version": r.get("icd_version"), "variant": variant, "src": "medico-letter"}))
    by = {k: _stratified_sample(v, sample.get(k, len(v)), seed + i) for i, (k, v) in enumerate(pools.items())}
    for k in pools:
        consort[f"pool_{k}"] = len(pools[k])
        consort[f"n_{k}"] = len(by[k])
    consort["patients_overlapping_splits"] = splits.overlap(by, "patient_id")
    return by, consort
