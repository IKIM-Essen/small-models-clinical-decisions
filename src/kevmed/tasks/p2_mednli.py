"""P2 MedNLI v1.0.0 (PhysioNet, credentialed): clinical NLI as a Kev `choice` over three relations
. Published train / dev / test files. Never prints rows."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

from .. import physionet, schema

TASK, KEV_TYPE = "p2", "choice"
OPTIONS = {"entailment": "The hypothesis follows from the premise",
           "contradiction": "The hypothesis contradicts the premise",
           "neutral": "The premise neither supports nor contradicts the hypothesis"}
INSTRUCTIONS = "Given the clinical premise, what is the relation of the hypothesis to it?"
FILES = {"train": "mli_train_v1.jsonl", "dev": "mli_dev_v1.jsonl", "test": "mli_test_v1.jsonl"}


def build(src: Path, seed: int, skip_bad: bool = False) -> tuple[dict[str, list[dict]], dict]:
    root = Path(src) / physionet.LAYOUT["mednli"]
    consort = OrderedDict(rows_in_source=0, dropped_unknown_label=0, dropped_empty=0)
    by = {}
    for split, fname in FILES.items():
        out = []
        with (root / fname).open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                consort["rows_in_source"] += 1
                pid = str(row["pairID"])
                label = row.get("gold_label")
                if label not in OPTIONS:
                    if not skip_bad:
                        raise ValueError(f"pairID {pid}: unknown gold_label")
                    consort["dropped_unknown_label"] += 1
                    continue
                if not (row.get("sentence1") or "").strip() or not (row.get("sentence2") or "").strip():
                    consort["dropped_empty"] += 1
                    continue
                out.append(schema.record(f"mednli-{pid}", {"premise": row["sentence1"], "hypothesis": row["sentence2"]},
                                         {"relation": schema.choice(INSTRUCTIONS, OPTIONS, label)},
                                         {"pair_id": pid, "group": pid, "src": "mednli"}))
        by[split] = out
        consort[f"n_{split}"] = len(out)
    return by, consort
