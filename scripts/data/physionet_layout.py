"""Arrange downloaded PhysioNet files into the kevmed.physionet.LAYOUT directories (runner pod only; the root must be
under $KEV_RESTRICTED_ROOT). Moves files, unzips the two archives, prints file counts and byte sizes only.
  uv run scripts/data/physionet_layout.py --incoming <download dir> --root data/public/physionet/restricted/physionet
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

from kevmed import physionet as pn, splits

# incoming file name -> LAYOUT key (plain files are moved to the LAYOUT path)
FILES = {
    "mli_train_v1.jsonl": ("mednli", "mli_train_v1.jsonl"), "mli_dev_v1.jsonl": ("mednli", "mli_dev_v1.jsonl"), "mli_test_v1.jsonl": ("mednli", "mli_test_v1.jsonl"),
    "edstays.csv.gz": ("ed_edstays", None), "triage.csv.gz": ("ed_triage", None), "patients.csv.gz": ("iv_patients", None),
    "NOTEEVENTS.csv.gz": ("iii_noteevents", None),
    "mimic-cxr-2.0.0-split.csv.gz": ("mimic_cxr_split", None), "mimic-cxr-2.0.0-chexpert.csv.gz": ("mimic_cxr_chexpert", None),
}
REPORTS_ZIP = "mimic-cxr-reports.zip"                       # members files/p1x/p<subject>/s<study>.txt -> mimic_cxr_reports
PHENO_ZIP_PREFIX = "phenotype-annotations-for-patient-notes"  # member .../ACTdb102003.csv -> phenotypes


def _size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size


def _count(p: Path) -> int:
    return sum(1 for f in p.rglob("*") if f.is_file()) if p.is_dir() else 1


def arrange(incoming: Path, root: Path) -> dict:
    incoming, root = Path(incoming), Path(root)
    report: dict[str, dict] = {}
    for name, (key, member) in FILES.items():
        src = incoming / name
        if not src.is_file():
            report[key] = {"missing": name}
            continue
        target = root / pn.LAYOUT[key]
        if member:
            target = target / member
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(target))
        report.setdefault(key, {"files": 0, "bytes": 0})
        report[key]["files"] += 1
        report[key]["bytes"] += target.stat().st_size
    src = incoming / REPORTS_ZIP
    if src.is_file():
        base = root / pn.LAYOUT["mimic_cxr_reports"]        # ".../mimic-cxr-2.1.0/files"
        with zipfile.ZipFile(src) as z:
            members = [m for m in z.namelist() if m.startswith("files/") and not m.endswith("/")]
            for m in members:
                out = base / Path(m).relative_to("files")
                out.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as fin, out.open("wb") as fout:
                    shutil.copyfileobj(fin, fout)
        src.unlink()
        report["mimic_cxr_reports"] = {"files": _count(base), "bytes": _size(base)}
    else:
        report["mimic_cxr_reports"] = {"missing": REPORTS_ZIP}
    pheno = next((p for p in incoming.glob("*.zip") if p.name.startswith(PHENO_ZIP_PREFIX)), None)
    if pheno:
        target = root / pn.LAYOUT["phenotypes"]
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(pheno) as z:
            csvs = [m for m in z.namelist() if m.endswith(".csv")]
            if len(csvs) != 1:
                raise RuntimeError(f"phenotype zip: expected one csv member, found {len(csvs)}")
            with z.open(csvs[0]) as fin, target.open("wb") as fout:
                shutil.copyfileobj(fin, fout)
        pheno.unlink()
        report["phenotypes"] = {"files": 1, "bytes": target.stat().st_size, "member": Path(csvs[0]).name}
    else:
        report["phenotypes"] = {"missing": "phenotype zip"}
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--incoming", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    a = ap.parse_args(argv)
    if not splits.under_restricted_root(a.root):
        print(f"--root must be under $KEV_RESTRICTED_ROOT ({splits.restricted_root()})", file=sys.stderr)
        return 4
    report = arrange(a.incoming, a.root)
    print(json.dumps(report, indent=1))
    missing = [k for k, v in report.items() if "missing" in v]
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
