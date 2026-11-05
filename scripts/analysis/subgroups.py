# /// script
# requires-python = ">=3.11"
# ///
"""Post hoc analysis:
accuracy and macro-F1 by age band and by sex on the test splits of P3 (triage acuity) and I1 (length of stay).

Age and sex come from each test record's state, as in scripts/analysis/characteristics.py. Rows: Kev-4B 3k and LoRA-SFT 3k
(means over the three seeds) and the better frontier model in its best-of-four mode (the H2 choice). Macro-F1 within a
subgroup averages over the classes present in it. Cells with fewer than 10 records are suppressed. Aggregates only.
Output: runs/subgroups.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import hypotheses as hy  # noqa: E402
import stats as st  # noqa: E402

TASKS = ("p3", "i1")
MIN_N = 10
AGE_BANDS = ((0, 18, "0-17"), (18, 40, "18-39"), (40, 60, "40-59"), (60, 80, "60-79"), (80, 200, "80+"))


def age_band(age) -> str:
    try:
        a = float(age)
    except (TypeError, ValueError):
        return "missing"
    for lo, hi, name in AGE_BANDS:
        if lo <= a < hi:
            return name
    return "missing"


def sex_group(sex) -> str:
    s = str(sex or "").strip().lower()
    return {"f": "female", "female": "female", "m": "male", "male": "male"}.get(s, "other or unknown")


def demographics(path: Path) -> dict[str, tuple[str, str]]:
    out = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rid = r.get("_meta", {}).get("id") or r.get("id")
                state = r.get("state") if isinstance(r.get("state"), dict) else {}
                out[str(rid)] = (age_band(state.get("age")), sex_group(state.get("sex")))
    return out


def cell(seed_rows: list[list[dict]], ids: set[str]) -> dict:
    sub = [[r for r in rows if r["id"] in ids] for rows in seed_rows]
    n = len({r["id"] for r in sub[0]})
    if n < MIN_N:
        return {"n": n, "suppressed": True}
    return {"n": n, "accuracy": float(np.mean([st.accuracy(r) for r in sub])), "macro_f1": float(np.mean([st.macro_f1(r) for r in sub]))}


def main():
    frozen = json.loads((ROOT / "configs" / "frozen.json").read_text())
    h2 = json.loads((ROOT / "runs" / "hypotheses-test.json").read_text())["H2"]["models"]
    long = {v: k for k, v in hy.FRONTIER.items()}
    out = {"note": __doc__.split("\n\n")[0], "min_n": MIN_N, "tasks": {}}
    for t in TASKS:
        demo = demographics(ROOT / "splits" / t / "test.jsonl")
        short = max(h2, key=lambda s: h2[s]["per_task"][t]["frontier"])
        mode = h2[short]["per_task"][t]["frontier_mode"]
        named = {"kev": hy.gather(ROOT, frozen, "kev4b-3k", t, "test"), "sft": hy.gather(ROOT, frozen, "sft4b-3k", t, "test"),
                 "frontier": [hy.frontier_rows(ROOT, long[short], mode, t)]}
        aligned, dropped = hy.align(named)
        groups: dict[str, dict[str, set]] = {"age": {}, "sex": {}}
        for rid in {r["id"] for r in aligned["kev"][0]}:
            a, s = demo.get(str(rid), ("missing", "other or unknown"))
            groups["age"].setdefault(a, set()).add(rid)
            groups["sex"].setdefault(s, set()).add(rid)
        res = {"frontier": {"model": long[short], "mode": mode}, "dropped_pairs": dropped, "all": {}, "age": {}, "sex": {}}
        every = {r["id"] for r in aligned["kev"][0]}
        for name, seeds in aligned.items():
            res["all"][name] = cell(seeds, every)
        for dim, gs in groups.items():
            for g, ids in sorted(gs.items()):
                res[dim][g] = {name: cell(seeds, ids) for name, seeds in aligned.items()}
        out["tasks"][t] = res
        print(t, short, mode, {d: {g: v["kev"]["n"] for g, v in res[d].items()} for d in ("age", "sex")}, flush=True)
    (ROOT / "runs" / "subgroups.json").write_text(json.dumps(out, indent=1) + "\n")
    print("wrote runs/subgroups.json")


if __name__ == "__main__":
    main()
