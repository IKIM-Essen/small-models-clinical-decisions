"""SYNTHETIC PhysioNet-shaped fixtures (no real rows). Run once: uv run python tests/fixtures/make_physionet.py
Every text field says SYNTHETIC. Layout mirrors kevmed.physionet.LAYOUT under tests/fixtures/physionet/."""
import csv
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).parent / "physionet"


def w_csv(rel, header, rows):
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def mednli():
    labels = ["entailment", "contradiction", "neutral"]
    for split, n in (("train", 9), ("dev", 3), ("test", 3)):
        p = ROOT / "mednli-1.0.0" / f"mli_{split}_v1.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            for i in range(n):
                lab = labels[i % 3] if not (split == "train" and i == 8) else "unknown"
                f.write(json.dumps({"pairID": f"{split}-{i:03d}", "sentence1": f"SYNTHETIC premise {split} {i}.",
                                    "sentence2": f"SYNTHETIC hypothesis {split} {i}.", "gold_label": lab,
                                    "sentence1_parse": "", "sentence2_parse": ""}) + "\n")


def mimic_iv_ed():
    dispositions = ["ADMITTED", "HOME", "HOME", "ADMITTED", "ELOPED", "EXPIRED", "LEFT WITHOUT BEING SEEN",
                    "LEFT AGAINST MEDICAL ADVICE", "TRANSFER", "OTHER", "HOME", "ADMITTED"] * 5
    edstays, triage, patients = [], [], []
    for i in range(60):
        subject = f"1{i // 2:07d}"          # two stays per patient
        stay = f"3{i:07d}"
        edstays.append([subject, "" if i % 3 else f"2{i:07d}", stay, f"2150-0{1 + i % 9}-1{i % 9} 1{i % 9}:00:00",
                        f"2150-0{1 + i % 9}-1{i % 9} 2{i % 3}:00:00", "F" if i % 2 else "M", "WHITE",
                        ["AMBULANCE", "WALK IN", "UNKNOWN"][i % 3], dispositions[i]])
        if i != 59:                          # one stay without a triage row
            acuity = "" if i == 57 else str(1 + i % 5)       # one missing acuity on a stay that survives the joins
            triage.append([subject, stay, "98.6" if i % 4 else "", "88", "18", "97", "120", "80",
                           str(i % 10), acuity, f"SYNTHETIC chief complaint {i}"])
    for i in range(30):
        if i != 29:                          # one patient missing from patients.csv
            patients.append([f"1{i:07d}", "F" if i % 2 else "M", str(20 + i * 2) if i < 28 else "91", "2148", "2017 - 2019", ""])
    w_csv("mimic-iv-ed-2.2/ed/edstays.csv.gz", ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "gender", "race", "arrival_transport", "disposition"], edstays)
    w_csv("mimic-iv-ed-2.2/ed/triage.csv.gz", ["subject_id", "stay_id", "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity", "chiefcomplaint"], triage)
    w_csv("mimic-iv-3.1/hosp/patients.csv.gz", ["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"], patients)


CXR_COLS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
            "Lung Opacity", "Pleural Effusion", "Pneumonia", "Pneumothorax", "Pleural Other", "Support Devices", "No Finding"]


def mimic_cxr():
    subjects = ["10000001", "10000002", "10000003", "10000004"]
    studies = [(subjects[0], "50000001"), (subjects[0], "50000002"), (subjects[1], "50000003"), (subjects[1], "50000004"),
               (subjects[2], "50000005"), (subjects[2], "50000006"), (subjects[3], "50000007")]   # the official split is patient-level
    labels, split, dicom = [], [], 0
    for k, (subj, study) in enumerate(studies):
        p = ROOT / "mimic-cxr-2.1.0" / "files" / f"p{subj[:2]}" / f"p{subj}" / f"s{study}.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"                                 FINAL REPORT\n SYNTHETIC EXAMINATION {k}:  CHEST\n\n FINDINGS:  SYNTHETIC findings {k}.\n\n IMPRESSION:  SYNTHETIC impression {k}.\n")
        if k == 0:
            row = ["1.0", "0.0", "", "-1.0"] + [""] * 9 + ["0.0"]          # the coding test study
        elif k == 5:
            row = ["-1.0"] * 14                                              # all uncertain -> dropped
        else:
            row = [["1.0", "0.0", ""][(k + j) % 3] for j in range(14)]
        if k != 4:                                                            # study 5 missing from chexpert -> dropped
            labels.append([subj, study] + row)
        for _ in range(2):
            dicom += 1
            split.append([f"d{dicom:06d}", study, subj, ["train", "train", "validate", "validate", "test", "test", "test"][k]])
    w_csv("mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz", ["subject_id", "study_id"] + CXR_COLS, labels)
    w_csv("mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-split.csv.gz", ["dicom_id", "study_id", "subject_id", "split"], split)


PHENO_COLS = ["Advanced.Cancer", "Advanced.Heart.Disease", "Advanced.Lung.Disease", "Alcohol.Abuse", "Chronic.Neurological.Dystrophies",
              "Chronic.Pain.Fibromyalgia", "Dementia", "Depression", "Developmental.Delay.Retardation", "Non.Adherence", "None",
              "Obesity", "Other.Substance.Abuse", "Schizophrenia.and.other.Psychiatric.Disorders", "Unsure"]


def phenotypes():
    rows, notes = [], []
    for i in range(12):
        row_id, subj, hadm = str(700000 + i), f"9{i // 2:04d}", str(800000 + i)
        base = [1 if (i + j) % 4 == 0 else 0 for j in range(14)]          # 14 phenotype flags incl. None at index 10
        base[10] = 0
        if i == 9:
            base = [0] * 14; base[10] = 1                                  # None by both annotators -> all false
        a = base + [0]
        b = list(base) + [0]
        if i == 10:
            b[0] = 1 - b[0]                                                # disagreement -> dropped
        if i == 11:
            a[14] = 1                                                      # Unsure by one operator -> dropped
        rows.append([row_id, subj, hadm, "1", "op_a"] + a)
        if i != 8:                                                         # note 8 has one annotator only -> dropped
            rows.append([row_id, subj, hadm, "1", "op_b"] + b)
        notes.append([row_id, subj, hadm, "2150-01-01", "", "", "Discharge summary", "Report", "", "", f"SYNTHETIC discharge summary {i}\n\nHospital course: SYNTHETIC."])
    notes.append(["799999", "99999", "899999", "2150-01-01", "", "", "Nursing", "", "", "", "SYNTHETIC unrelated note"])
    w_csv("phenotype-annotations-mimic-1.20.03/ACTdb102003.csv", ["ROW_ID", "SUBJECT_ID", "HADM_ID", "BATCH.ID", "OPERATOR"] + PHENO_COLS, rows)
    w_csv("mimic-iii-1.4/NOTEEVENTS.csv.gz", ["ROW_ID", "SUBJECT_ID", "HADM_ID", "CHARTDATE", "CHARTTIME", "STORETIME", "CATEGORY", "DESCRIPTION", "CGID", "ISERROR", "TEXT"], notes)


if __name__ == "__main__":
    mednli()
    mimic_iv_ed()
    mimic_cxr()
    phenotypes()
    print("wrote", ROOT)
