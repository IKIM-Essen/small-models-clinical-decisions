"""SYNTHETIC internal raw rows (the shape fmx_extract.py writes), no real patients. Run once:
  uv run python tests/fixtures/make_internal.py"""
import json
from pathlib import Path

ROOT = Path(__file__).parent / "internal" / "raw"


def w(rel, rows):
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def i3():
    rows = []
    codes = [["8140/3~"], ["8000/0"], ["8140/6~"], ["8500/2", "8500/3~"], ["8140/3", "8140/6"], ["9999/9"], ["abcd"], ["8010/1~"]]
    for i in range(16):
        year = [2019, 2021, 2024, 2025, 2026][i % 5]
        rows.append({"report_id": f"R{i:04d}", "patient_id": f"P{i // 2:03d}", "year": year, "issued": f"{year}-03-0{1 + i % 9}",
                     "icdo3_codes": codes[i % len(codes)],
                     "markdown": ("SYNTHETIC pathology report %d. " % i) * (12 if i != 15 else 1)})
    w("i3/rows.jsonl", rows)


def i1():
    rows, labs, hist = [], [], []
    for i in range(24):
        year = [2021, 2022, 2023, 2024, 2025, 2026][i % 6]
        admit = f"{year}-05-1{i % 9}T08:00:00+02:00"
        los = [0.5, 2, 3, 5, 9, 1][i % 6]
        discharge = None if i == 20 else f"{year}-05-{10 + i % 9 + int(los):02d}T12:00:00+02:00" if i != 21 else f"{year}-05-01T00:00:00+02:00"
        rows.append({"case_id": f"C{i:04d}", "patient_id": f"P{i // 3:03d}", "admit": admit, "discharge": discharge, "year": year,
                     "department": "SYNTHETIC department", "age": 30 + i, "sex": "female" if i % 2 else "male",
                     "auf_codes": None if i == 22 else ["I21.0"], "auf_texts": None if i == 22 else ["SYNTHETIC Akuter Myokardinfarkt I21.0"],
                     "auf_alpha_only": i == 23})
        labs.append({"case_id": f"C{i:04d}", "name": "SYNTHETIC Kalium", "code": "K", "value": 4.1, "unit": "mmol/L", "taken": admit})
        hist.append({"case_id": f"C{i:04d}", "prior_codes": ["E11.9"], "prior_texts": ["SYNTHETIC Diabetes E11.9"]})
    w("i1/rows.jsonl", rows); w("i1/labs.jsonl", labs); w("i1/history.jsonl", hist)


def i2():
    rows = []
    chapters = ["C34.1", "I21.0", "J18.9", "K80.2", "E11.9", "C50.9", "I50.1", "N18.5"]
    for i in range(40):
        year = [2023, 2024, 2025, 2026][i % 4]
        code = chapters[i % len(chapters)]
        rows.append({"case_id": f"C{i:04d}", "patient_id": f"P{i:03d}", "year": year, "drg_code": code, "icd_version": str(year),
                     "doc_id": f"D{i:04d}", "doc_date": f"{year}-06-0{1 + i % 9}T10:00:00+02:00", "pdf_url": f"https://ship.example/app/docs/cla/{i}"})
        md = ROOT / "i2" / "ocr" / f"D{i:04d}.md"
        md.parent.mkdir(parents=True, exist_ok=True)
        if i == 38:
            continue                                                     # no OCR output
        body = "" if i == 39 else ("## Anamnese\n" + "SYNTHETIC Verlauf. " * 20 + f"\n\n## Diagnosen\nSYNTHETIC Hauptdiagnose {code}\n\n## Therapie\n" + "SYNTHETIC Therapie. " * 5)
        md.write_text(body)
    rows.append(dict(rows[0], doc_id="D0100", doc_date="2023-06-09T10:00:00+02:00", pdf_url="https://ship.example/app/docs/cla/100"))   # second, later letter for C0000
    (ROOT / "i2" / "ocr" / "D0100.md").write_text("## Diagnosen\nSYNTHETIC later letter C34.1\n## Therapie\n" + "SYNTHETIC text. " * 30)
    rows.append(dict(rows[1], case_id="C0101", doc_id="D0101", drg_code="ZZ9"))                                                          # unmapped code (X99 would be chapter XX)
    (ROOT / "i2" / "ocr" / "D0101.md").write_text("SYNTHETIC text. " * 30)
    w("i2/rows.jsonl", rows)


def i4():
    rows = []
    for i in range(24):
        year = [2021, 2022, 2023, 2024, 2025, 2026][i % 6]
        discharge = f"{year}-05-{12 + i % 9:02d}T12:00:00+02:00"
        nxt = None if i % 3 else f"{year}-06-{1 + i % 9:02d}T09:00:00+02:00"      # 20 days later -> readmitted
        if i == 6:
            nxt = f"{year}-08-01T09:00:00+02:00"                                   # > 30 days -> not readmitted
        rows.append({"case_id": f"C{i:04d}", "patient_id": f"P{i // 3:03d}", "discharge": discharge, "next_admit": nxt})
    w("i4/rows.jsonl", rows)


if __name__ == "__main__":
    i3()
    i1()
    i2()
    i4()
    print("wrote", ROOT)
