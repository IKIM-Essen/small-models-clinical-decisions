# SYNTHETIC fixtures

Every file under this directory is invented by `tests/fixtures/make_physionet.py` and mirrors the PhysioNet file
layout in `kevmed.physionet.LAYOUT` (column names, value coding, one edge case per builder rule). No row comes from
MIMIC, MedNLI or the phenotype annotations. Files without a free-text field (the annotation CSV, the split CSV) carry
no marker inside; this README is their marker.
