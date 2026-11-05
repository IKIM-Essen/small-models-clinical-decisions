# /// script
# requires-python = ">=3.11"
# ///
"""Label conventions for the frontier 'definitions' arm: per task, the
local label rule in words plus the label frequencies on the 3k training subset (the same records the fine-tuned rows and
the few-shot examples come from). Writes scripts/frontier/label_definitions.json; scripts/frontier/serve.py appends the
task's block to the system prompt when FRONTIER_DEFS=1. Never prints row content."""
from __future__ import annotations

import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = {"p1": "data/derived/p1-3k/train.jsonl", "p2": "data/derived/p2-3k/train.jsonl", "p3": "data/derived/p3-3k/train.jsonl",
         "i1": "data/derived/i1-3k/train.jsonl", "i2": "data/derived/i2-sf7400/train.jsonl", "i3": "data/derived/i3-3k/train.jsonl"}
RULES = {
    "p1": ("The labels were produced by the CheXpert rule-based labeler applied to this same report text. A finding is "
           "'yes' only when the report states it as present; a finding that is not mentioned counts as 'no'; negated "
           "mentions count as 'no'; 'no finding' is 'yes' only when no abnormal finding is mentioned at all."),
    "p2": ("MedNLI: 'entailment' when the hypothesis must be true given the premise, 'contradiction' when it cannot be "
           "true given the premise, 'neutral' when the premise does not settle it. The three labels are equally frequent."),
    "p3": ("The label is the triage nurse's Emergency Severity Index at an emergency department visit (MIMIC-IV-ED). "
           "ESI 1 means immediate life-saving intervention; ESI 2 high-risk, confused, lethargic, severe pain or distress, "
           "or unstable vital signs; ESI 3 two or more resources expected; ESI 4 one resource; ESI 5 none. In this "
           "department most visits are ESI 3, and ESI 5 is almost never assigned."),
    "i1": ("The label is the length of the inpatient stay that followed this admission, from the actual encounter period: "
           "'1 day or less', '2 to 3 days', '4 to 7 days', 'more than 7 days'. Only admission-time information is shown. "
           "Stays of 2 to 7 days are the most common; stays of one day or less are rare."),
    "i2": ("The label is the ICD-10-GM chapter of the single principal diagnosis that was coded for billing (DRG) at the "
           "end of this stay, mapped by code range: chapter II (C00-D48) includes benign and uncertain neoplasms "
           "D10-D48; chapter III starts at D50; chapter VII is the eye (H00-H59) and chapter VIII the ear (H60-H95); "
           "chapter XIX covers injuries and poisoning (S00-T98). Neoplasms (II) and circulatory diseases (IX) together "
           "make up over half of the stays."),
    "i3": ("The label is the ICD-O-3 behaviour digit of the pathologist's coded conclusion for this report: behaviour /3 "
           "(malignant, primary), /6 (metastatic) and /9 count as malignant; /0 (benign), /1 (uncertain) and /2 (in situ) "
           "count as not malignant. 'Metastatic' is asked only for malignant reports and means behaviour /6. Most coded "
           "reports in this archive are malignant."),
}


def main():
    out = {}
    for t, path in TRAIN.items():
        counts = collections.defaultdict(collections.Counter)
        n = 0
        for line in open(ROOT / path):
            if not line.strip():
                continue
            r = json.loads(line); n += 1
            for q, v in r["questions"].items():
                y = v.get("label")
                if v["type"] == "noul":
                    k = "yes" if y else "no"
                elif v["type"] == "score":
                    k = str(v["criteria"][int(y)])          # the level's name, not its index
                else:
                    k = str(y)
                counts[q][k] += 1
        rates = {q: {k: round(v / sum(c.values()), 3) for k, v in sorted(c.items(), key=lambda kv: -kv[1])} for q, c in counts.items()}
        text = RULES[t] + " Label frequencies in the training sample: " + "; ".join(
            f"{q}: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in r.items()) for q, r in rates.items()) + "."
        out[t] = {"rule": RULES[t], "train_records": n, "rates": rates, "text": text}
    (ROOT / "scripts" / "frontier" / "label_definitions.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    for t in out:
        print(t, len(out[t]["text"]), "chars")


if __name__ == "__main__":
    main()
