import pytest

from kevmed import icd


@pytest.mark.parametrize("code,chapter", [
    ("A00", 1), ("B99.9", 1), ("C34.1", 2), ("D48", 2), ("D48.9", 2), ("D50", 3), ("D89", 3),
    ("E11.9", 4), ("F32", 5), ("G40", 6), ("H25", 7), ("H59", 7), ("H60", 8), ("H95", 8),
    ("I21.0", 9), ("J18", 10), ("K80", 11), ("L40", 12), ("M54", 13), ("N18", 14), ("O80", 15),
    ("P07", 16), ("Q21", 17), ("R50", 18), ("S72.0", 19), ("T98", 19), ("V01", 20), ("Y84", 20),
    ("Z38", 21), ("U07.1", 22), ("U99", 22),
])
def test_icd10_chapter_boundaries(code, chapter):
    assert icd.icd10_chapter(code) == chapter


@pytest.mark.parametrize("bad", ["", "X", "1A2", "Y85", "H96", "T99", "d50"])
def test_icd10_chapter_none_for_unmapped(bad):
    assert icd.icd10_chapter(bad) is None


def test_chapter_options_has_22_roman_keys():
    opts = icd.chapter_options()
    assert len(opts) == 22 and list(opts)[:3] == ["I", "II", "III"] and "A00-B99" in opts["I"]


@pytest.mark.parametrize("code,beh", [("8140/3", 3), ("8140/3~", 3), ("8000/1", 1), ("8500/2~", 2), ("8140/6", 6), ("8140/9", 9)])
def test_icdo3_behaviour(code, beh):
    assert icd.icdo3_behaviour(code) == beh


@pytest.mark.parametrize("bad", ["8140", "8140/", "814/3", "abcd/3", "", "8140/3/1"])
def test_icdo3_behaviour_none(bad):
    assert icd.icdo3_behaviour(bad) is None


def test_malignant_and_metastatic():
    assert [icd.is_malignant(b) for b in (0, 1, 2, 3, 6, 9)] == [False, False, False, True, True, True]
    assert icd.is_metastatic(6) is True and icd.is_metastatic(3) is False and icd.is_metastatic(0) is None


def test_strip_icd_codes():
    text, n = icd.strip_icd_codes("Diagnose: I21.0 STEMI, Nebendiagnose E11.9; Vitamin B12 normal. Kein Code x99.")
    assert "I21.0" not in text and "E11.9" not in text and n == 3   # B12 matches the spec pattern too
    assert "STEMI" in text and "x99" in text
