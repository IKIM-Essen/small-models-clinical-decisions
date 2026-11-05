"""Per-task, per-split characteristics -> runs/characteristics.json.

Aggregate counts only: records, distinct groups (patients or subjects, as each builder sets `_meta.group`), label
counts per question, year range of `_meta.year` where present, age (median, quartiles, missing) and sex counts where
the state carries them (I1 `state.age`/`state.sex`, P3 the same from the ED state). Reads the frozen split files and
the 3k training subsets on the pod; never prints or writes row content."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median

TASKS = ("p1", "p2", "p3", "i1", "i2", "i3")


def _quartiles(values: list[float]) -> dict:
    v = sorted(values)
    n = len(v)
    if n == 0:
        return {"n": 0}
    q = lambda p: v[min(n - 1, int(round(p * (n - 1))))]
    return {"n": n, "median": median(v), "q1": q(0.25), "q3": q(0.75), "min": v[0], "max": v[-1]}


def summarize(path: Path) -> dict:
    n = 0
    groups: set[str] = set()
    labels: dict[str, Counter] = {}
    years: Counter = Counter()
    ages: list[float] = []
    age_missing = 0
    sex: Counter = Counter()
    has_demo = False
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            meta = r.get("_meta", {})
            g = meta.get("group")
            if g is not None:
                groups.add(str(g))
            if meta.get("year") is not None:
                years[int(meta["year"])] += 1
            for qid, q in r.get("questions", {}).items():
                lab = q.get("label")
                if q.get("type") == "score":
                    lab = q["criteria"][lab] if isinstance(lab, int) and 0 <= lab < len(q["criteria"]) else lab
                labels.setdefault(qid, Counter())[str(lab)] += 1
            st = r.get("state")
            if isinstance(st, dict) and ("age" in st or "sex" in st):
                has_demo = True
                a = st.get("age")
                try:
                    a = float(a)
                except (TypeError, ValueError):
                    a = None
                if a is None:
                    age_missing += 1
                else:
                    ages.append(a)
                sex[str(st.get("sex") or "unknown").lower()] += 1
    out = {
        "n_records": n,
        "n_groups": len(groups),
        "labels": {qid: dict(sorted(c.items())) for qid, c in labels.items()},
    }
    if years:
        out["years"] = {"min": min(years), "max": max(years), "counts": dict(sorted(years.items()))}
    if has_demo:
        out["age"] = {**_quartiles(ages), "missing": age_missing}
        out["sex"] = dict(sorted(sex.items()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="runs/characteristics.json")
    args = ap.parse_args()
    root = Path(args.root)
    result: dict = {}
    for task in TASKS:
        files = {s: root / "splits" / task / f"{s}.jsonl" for s in ("train", "dev", "test")}
        sub = root / "data" / "derived" / f"{task}-3k" / "train.jsonl"
        if sub.exists():
            files["subset"] = sub
        result[task] = {}
        for split, p in files.items():
            if not p.exists():
                result[task][split] = None
                continue
            result[task][split] = summarize(p)
            s = result[task][split]
            print(task, split, "records", s["n_records"], "groups", s["n_groups"],
                  "age" if "age" in s else "", s.get("age", {}).get("median", ""), flush=True)
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
