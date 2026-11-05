# /// script
# requires-python = ">=3.11"
# ///
"""Sensitivity analyses: (1) the frontier 'definitions' arm: label conventions and training-label frequencies in the system prompt, scored on the same 500 development
records per task as the plain zero-shot arm, non-thinking, both models, 0- and 3-shot; paired with the plain zero-shot
arm on the same records, and with fine-tuned Kev-4B (3k, seed 0) restricted to those records; since 2026-10-27 also
with thinking, against the plain thinking prompt at the same shot count. (2) I3 without
code-bearing reports and unseen patients: Kev-4B 3k and LoRA-SFT 3k retrained on the code-free subset and
scored on the code-free, unseen-patient test records; compared with the original 3k models restricted to the same test
records. Output: runs/sensitivity-extra.json (aggregates only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import stats as st  # noqa: E402

TASKS = ("p1", "p2", "p3", "i1", "i2", "i3")


def rows_of(d):
    p = ROOT / d / "rows.json"
    return st.load_rows(ROOT / d) if p.exists() else None


def metrics(rows):
    a = st.arrays(rows)
    return {"accuracy": st.accuracy(a), "macro_f1": st.macro_f1(a), "automation_at_5pct": st.automation(a), "n": len(rows)}


def restrict(rows, keys):
    index = {st.key(r): r for r in rows}
    return [index[k] for k in keys if k in index]


def main():
    out = {"note": __doc__.split("\n\n")[0], "definitions": {}, "definitions_think": {}, "i3_nocode": {}}
    frozen = json.loads((ROOT / "configs" / "frozen.json").read_text())["entries"]
    for t in TASKS:
        cell = {}
        plain = rows_of(f"runs/frontier-deepseek-v4.1-flash-nothink-k0-{t}-dev500")
        keys = [st.key(r) for r in plain] if plain else None
        for model in ("deepseek-v4.1-flash", "kimi-k3"):
            p0 = rows_of(f"runs/frontier-{model}-nothink-k0-{t}-dev500")
            for k in ("0", "3"):
                d = rows_of(f"runs/frontier-{model}-nothink-k{k}-defs-{t}-dev500")
                if d is None or p0 is None:
                    continue
                shared, _ = st.common({"plain": p0, "defs": d})
                cell[f"{model}/k{k}"] = {"plain_k0": metrics(shared["plain"]), "defs": metrics(shared["defs"]),
                                         "accuracy_gain": metrics(shared["defs"])["accuracy"] - metrics(shared["plain"])["accuracy"]}
                # round 2: the matched-shot contrast, definitions minus the plain prompt at the same k
                pk = p0 if k == "0" else rows_of(f"runs/frontier-{model}-nothink-k3-{t}-dev500")
                if pk is not None:
                    shared_k, _ = st.common({"plain": pk, "defs": d})
                    cell[f"{model}/k{k}"]["plain_same_k"] = metrics(shared_k["plain"])
                    cell[f"{model}/k{k}"]["matched_gain"] = metrics(shared_k["defs"])["accuracy"] - metrics(shared_k["plain"])["accuracy"]
                    cell[f"{model}/k{k}"]["matched_gain_macro_f1"] = metrics(shared_k["defs"])["macro_f1"] - metrics(shared_k["plain"])["macro_f1"]
        # 2026-10-27: the same contrast with thinking, definitions minus the plain thinking prompt
        # at the same shot count on the shared records (rows without a parsed answer are dropped from both, counted in n)
        think = {}
        for model in ("deepseek-v4.1-flash", "kimi-k3"):
            for k in ("0", "3"):
                pk = rows_of(f"runs/frontier-{model}-think-k{k}-{t}-dev500")
                d = rows_of(f"runs/frontier-{model}-think-k{k}-defs-{t}-dev500")
                if pk is None or d is None:
                    continue
                sh, dropped = st.common({"plain": pk, "defs": d})
                mp, md = metrics(sh["plain"]), metrics(sh["defs"])
                think[f"{model}/k{k}"] = {"plain_same_k": mp, "defs": md, "dropped": dropped,
                                          "matched_gain": md["accuracy"] - mp["accuracy"],
                                          "matched_gain_macro_f1": md["macro_f1"] - mp["macro_f1"]}
        out.setdefault("definitions_think", {})[t] = think
        e = frozen.get(f"kev4b-3k/{t}/s0")
        if e and keys:
            kev = st.with_temperature(st.load_rows(ROOT / e["dev_dir"]), e["dev_temperature"])
            cell["kev4b-3k/s0"] = metrics(restrict(kev, keys))
        out["definitions"][t] = cell
    # I3 without code-bearing reports and unseen patients
    test_keys = None
    for row in ("kev4b", "sft4b"):
        new = rows_of(f"runs/{row}-i3-3k-nocode-s0-test")
        if new is None:
            continue
        test_keys = [st.key(r) for r in new]
        e = frozen[f"{row}-3k/i3/s0"]
        orig = st.with_temperature(st.load_rows(ROOT / e["test_out"]), e["dev_temperature"])
        dev_t = float(json.loads((ROOT / f"runs/{row}-i3-3k-nocode-s0-dev/calibration.json").read_text())["workload_temperature"])
        out["i3_nocode"][row] = {"retrained_nocode": metrics(st.with_temperature(new, dev_t)),
                                 "original_3k_same_records": metrics(restrict(orig, test_keys)),
                                 "original_3k_full_test": metrics(orig)}
    out["i3_nocode"]["counts"] = json.loads((ROOT / "data/derived/i3-3k-nocode/counts.json").read_text())
    gains = [v["matched_gain"] for c in out["definitions"].values() for v in c.values() if "matched_gain" in v]
    tgains = [v["matched_gain"] for c in out.get("definitions_think", {}).values() for v in c.values()]
    out["definitions_range"] = {"nothink": {"min": min(gains), "max": max(gains)} if gains else None,
                                "think": {"min": min(tgains), "max": max(tgains)} if tgains else None}
    (ROOT / "runs" / "sensitivity-extra.json").write_text(json.dumps(out, indent=1) + "\n")
    for t in TASKS:
        c = out["definitions"][t]
        print(t, {k: (round(v["plain_k0"]["accuracy"], 3), round(v["defs"]["accuracy"], 3)) for k, v in c.items() if "/k" in k}, "kev", round(c.get("kev4b-3k/s0", {}).get("accuracy", float("nan")), 3))
    for row, v in out["i3_nocode"].items():
        if row != "counts":
            print(row, {k: round(x["accuracy"], 3) for k, x in v.items()})


if __name__ == "__main__":
    main()
