from conftest import REPO
from kevmed.tasks import i2_letter as i2

FX = REPO / "tests/fixtures/internal/raw"
SMALL = {"train": 100, "dev": 100, "test": 100}


def test_one_letter_per_case_latest_wins():
    by, consort = i2.build(FX, seed=1, sample=SMALL)
    allr = {r["_meta"]["case_id"]: r for v in by.values() for r in v}
    assert allr["C0000"]["_meta"]["doc_id"] == "D0100" and consort["extra_letters"] == 1


def test_short_ocr_text_dropped():
    by, consort = i2.build(FX, seed=1, sample=SMALL)
    assert consort["dropped_no_ocr"] == 1 and consort["dropped_short_text"] == 1 and consort["dropped_unmapped_code"] == 1


def test_codes_stripped_and_chapter_label():
    by, consort = i2.build(FX, seed=1, sample=SMALL)
    r = next(x for v in by.values() for x in v if x["_meta"]["case_id"] == "C0001")
    assert r["questions"]["chapter"]["label"] == "IX" and "I21.0" not in r["state"] and consort["codes_stripped"] >= 30
    assert len(r["questions"]["chapter"]["criteria"]) == 22


def test_no_diagnosis_variant_removes_section_and_sampling_caps():
    by, consort = i2.build(FX, seed=1, sample={"train": 4, "dev": 2, "test": 2}, variant="no_diagnosis")
    assert (len(by["train"]), len(by["dev"]), len(by["test"])) == (4, 2, 2)
    assert all("Hauptdiagnose" not in r["state"] and "Therapie" in r["state"] and r["_meta"]["variant"] == "no_diagnosis" for v in by.values() for r in v)


def test_ledger_separates_not_selected_from_ocr_failures(tmp_path):
    import json
    import shutil
    root = tmp_path / "raw"
    shutil.copytree(FX, root)
    ocr = root / "i2" / "ocr"
    ledger = ocr / "ledger.jsonl"
    selected = {"D0000", "D0001", "D0002", "D0038", "D0100"}          # the pre-OCR sample; D0038 failed at the sidecar
    for md in ocr.glob("D*.md"):
        if md.stem not in selected:
            md.unlink()                                                # never sent to OCR
    with ledger.open("w") as f:
        for doc in sorted(selected):
            f.write(json.dumps({"doc_id": doc, "status": "failed" if doc == "D0038" else "ok", "chars": 500, "seconds": 1.0}) + "\n")
    by, consort = i2.build(root, seed=1, sample=SMALL)
    assert consort["not_selected"] > 30 and consort["dropped_no_ocr"] == 1
    assert sum(len(v) for v in by.values()) == 3            # D0100 (case C0000), D0001, D0002
