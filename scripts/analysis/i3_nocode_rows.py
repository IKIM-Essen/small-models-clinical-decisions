# /// script
# requires-python = ">=3.11"
# ///
"""Sensitivity analysis: every I3 test row restricted to the code-free, unseen-patient test records of
data/derived/i3-3k-nocode (the records scored by the retrained models), plus the retrained Kev-4B, LoRA-SFT and classical
rows on them. Accuracy, macro-F1 and automation at 5% (local rows at their dev-fitted temperature; frontier rows raw,
thinking rows accuracy and macro-F1 only). Output: runs/i3-nocode-rows.json (aggregates only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import stats as st  # noqa: E402

FRONTIER = [(m, k) for m in ("deepseek-v4.1-flash", "kimi-k3") for k in ("nothink-k0", "nothink-k3", "think-k0", "think-k3")]


def metrics(rows, automation=True):
    a = st.arrays(rows)
    out = {"accuracy": st.accuracy(a), "macro_f1": st.macro_f1(a), "n": len(rows)}
    if automation:
        out["automation_at_5pct"] = st.automation(a)
    return out


def restrict(rows, keys):
    index = {st.key(r): r for r in rows}
    return [index[k] for k in keys if k in index]


def main():
    frozen = json.loads((ROOT / "configs" / "frozen.json").read_text())["entries"]
    base = st.load_rows(ROOT / "runs/kev4b-i3-3k-nocode-s0-test")
    keys = [st.key(r) for r in base]
    out = {"note": __doc__.split("\n\n")[0], "n_records_nocode": len({k[0] for k in keys}), "n_decisions_nocode": len(keys), "rows": {}}
    # original rows restricted
    for name, e in frozen.items():
        if "/i3" not in name or not (ROOT / e["test_out"] / "rows.json").exists():
            continue
        rows = st.load_rows(ROOT / e["test_out"])
        if e.get("dev_temperature") is not None:
            rows = st.with_temperature(rows, e["dev_temperature"])
        out["rows"][name] = {"restricted": metrics(restrict(rows, keys)), "full": metrics(rows)}
        print(name, {k: round(v, 4) for k, v in out["rows"][name]["restricted"].items()}, flush=True)
    for m, k in FRONTIER:
        d = ROOT / f"runs/frontier-{m}-{k}-i3-test"
        if not (d / "rows.json").exists():
            continue
        rows = st.load_rows(d)
        auto = not k.startswith("think")
        out["rows"][f"{m}/{k}"] = {"restricted": metrics(restrict(rows, keys), auto), "full": metrics(rows, auto)}
        print(m, k, {kk: round(v, 4) for kk, v in out["rows"][f"{m}/{k}"]["restricted"].items()}, flush=True)
    # retrained rows on the code-free subset
    for row in ("kev4b", "sft4b", "classical"):
        d = ROOT / f"runs/{row}-i3-3k-nocode-s0-test"
        c = ROOT / f"runs/{row}-i3-3k-nocode-s0-dev/calibration.json"
        if not (d / "rows.json").exists():
            continue
        rows = st.load_rows(d)
        if c.exists():
            rows = st.with_temperature(rows, float(json.loads(c.read_text())["workload_temperature"]))
        out["rows"][f"{row}-3k-nocode/s0"] = {"restricted": metrics(restrict(rows, keys)), "full": metrics(rows)}
        print(row, "retrained", {k: round(v, 4) for k, v in out["rows"][f"{row}-3k-nocode/s0"]["restricted"].items()}, flush=True)
    (ROOT / "runs" / "i3-nocode-rows.json").write_text(json.dumps(out, indent=1) + "\n")
    print("wrote runs/i3-nocode-rows.json")


if __name__ == "__main__":
    main()
