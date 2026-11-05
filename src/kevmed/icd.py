"""ICD-10-GM chapter mapping by code range (not first letter: D and H span two chapters), ICD-O-3 behaviour
digit (with the trailing `~` FMX writes since 2017), and the ICD-code stripper for the I2 state text
."""
from __future__ import annotations

import re

# (number, roman, first code, last code, title). ICD-10-GM chapter XX is V01-Y84, XXII is U00-U99.
CHAPTERS = [
    (1, "I", "A00", "B99", "Certain infectious and parasitic diseases"),
    (2, "II", "C00", "D48", "Neoplasms"),
    (3, "III", "D50", "D89", "Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism"),
    (4, "IV", "E00", "E90", "Endocrine, nutritional and metabolic diseases"),
    (5, "V", "F00", "F99", "Mental and behavioural disorders"),
    (6, "VI", "G00", "G99", "Diseases of the nervous system"),
    (7, "VII", "H00", "H59", "Diseases of the eye and adnexa"),
    (8, "VIII", "H60", "H95", "Diseases of the ear and mastoid process"),
    (9, "IX", "I00", "I99", "Diseases of the circulatory system"),
    (10, "X", "J00", "J99", "Diseases of the respiratory system"),
    (11, "XI", "K00", "K93", "Diseases of the digestive system"),
    (12, "XII", "L00", "L99", "Diseases of the skin and subcutaneous tissue"),
    (13, "XIII", "M00", "M99", "Diseases of the musculoskeletal system and connective tissue"),
    (14, "XIV", "N00", "N99", "Diseases of the genitourinary system"),
    (15, "XV", "O00", "O99", "Pregnancy, childbirth and the puerperium"),
    (16, "XVI", "P00", "P96", "Certain conditions originating in the perinatal period"),
    (17, "XVII", "Q00", "Q99", "Congenital malformations, deformations and chromosomal abnormalities"),
    (18, "XVIII", "R00", "R99", "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified"),
    (19, "XIX", "S00", "T98", "Injury, poisoning and certain other consequences of external causes"),
    (20, "XX", "V01", "Y84", "External causes of morbidity and mortality"),
    (21, "XXI", "Z00", "Z99", "Factors influencing health status and contact with health services"),
    (22, "XXII", "U00", "U99", "Codes for special purposes"),
]

_CODE = re.compile(r"^([A-Z])(\d\d)(?:\.\d+)?$")
_ICDO = re.compile(r"^(\d{4})/(\d)~?$")
_INLINE = re.compile(r"\b[A-Z]\d\d(?:\.\d+)?\b")


def _key(code3: str) -> tuple[str, int]:
    return code3[0], int(code3[1:3])


def icd10_chapter(code: str) -> int | None:
    m = _CODE.match(code or "")
    if not m:
        return None
    k = (m.group(1), int(m.group(2)))
    for num, _, lo, hi, _ in CHAPTERS:
        if _key(lo) <= k <= _key(hi):
            return num
    return None


def chapter_options() -> dict[str, str]:
    return {roman: f"{title} ({lo}-{hi})" for _, roman, lo, hi, title in CHAPTERS}


def icdo3_behaviour(code: str) -> int | None:
    m = _ICDO.match((code or "").strip())
    return int(m.group(2)) if m else None


def is_malignant(behaviour: int) -> bool:
    return behaviour in (3, 6, 9)


def is_metastatic(behaviour: int) -> bool | None:
    return {6: True, 3: False}.get(behaviour)


def strip_icd_codes(text: str) -> tuple[str, int]:
    out, n = _INLINE.subn("", text)
    return out, n
