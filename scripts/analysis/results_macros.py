# /// script
# requires-python = ">=3.11"
# ///
"""Results -> runs/results/metrics.json.

Collects the hypothesis, latency and sensitivity results into one metrics file. Confidence intervals are written as
{lo, hi}."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RUN = "results"
TASK = {"p1": "POne", "p2": "PTwo", "p3": "PThree", "i1": "IOne", "i2": "ITwo", "i3": "IThree"}
MODEL = {"deepseek": "Deepseek", "kimi": "Kimi", "deepseek-v41-flash": "Deepseek", "kimi-k3": "Kimi"}
MODE = {"nothink-k0": "NoThinkZero", "nothink-k3": "NoThinkThree", "think-k0": "ThinkZero", "think-k3": "ThinkThree"}
# rows of the full test table (runs/test-rows-summary.json), dev-only descriptive rows, sizes
ROW = {"kev4b-3k": "KevThreeK", "sft4b-3k": "SftThreeK", "kev4b-full": "KevFull", "kev4b-zeroshot": "KevZero",
       "qwen4b-zeroshot-k0": "QwenZero", "qwen4b-zeroshot-k3": "QwenThree", "classical-3k": "ClassicalThreeK"}
METRIC = {"accuracy": ("Acc", "pct1"), "macro_f1": ("MacroF", ".3f"), "automation_at_5pct": ("Auto", "pct1")}
DEV_RUNS = {"ClefFlash": ("clef-flash-zeroshot-{t}-dev",), "Clef": ("clef-zeroshot-{t}-dev",),
            "Pplx": ("pplx-decider-zeroshot-{t}-dev",), "KevTwentySeven": ("kev-27b-zeroshot-{t}-dev",),
            "Laya": ("laya-zeroshot-{t}-dev", "laya-zeroshot-{t}"), "KevNine": ("kev9b-{t}-full-s0-dev",)}


def _ci_dicts(x):
    if isinstance(x, dict):
        return {k: ({"lo": v[0], "hi": v[1]} if k == "ci95" and isinstance(v, list) else _ci_dicts(v)) for k, v in x.items()}
    return x


def _ci(macros, name, key, fmt):
    macros[name] = {"run": RUN, "key": f"{key}.point", "fmt": fmt}
    macros[name + "Lo"] = {"run": RUN, "key": f"{key}.ci95.lo", "fmt": fmt}
    macros[name + "Hi"] = {"run": RUN, "key": f"{key}.ci95.hi", "fmt": fmt}


def _triple(v):
    """[accuracy, macro-F1, automation at 5%] as in the summaries; None stays absent (no value, no macro)."""
    return None if v is None else {k: x for k, x in zip(METRIC, v) if x is not None}


def load_extra(root: Path, test_rows: dict, frontier: dict, curves: dict | None) -> dict:
    """The inputs as metrics.json sections: test and frontier summaries, dev-only reports, sizes, automation at 1%."""
    root = Path(root)
    out = {"test_rows": {r: {t: _triple(v) for t, v in tasks.items() if v is not None} for r, tasks in test_rows.items()},
           "frontier_rows": {c.replace(".", ""): _triple(v) for c, v in frontier.items()}}
    out["dev_rows"] = {}
    for name, patterns in DEV_RUNS.items():
        for t in TASK:
            for pat in patterns:
                rep = root / "runs" / pat.format(t=t) / "report.json"
                if rep.exists():
                    c = json.loads(rep.read_text())["clean"]
                    # accuracy only: it does not depend on the temperature, and these dev-only rows have no
                    # test-protocol temperature (kev.benchmark reports them at the shipped temperature)
                    out["dev_rows"].setdefault(name, {})[t] = {"accuracy": c["acc"], "n_rows": c["n"], "run": pat.format(t=t)}
                    break
    out["sizes"] = {}
    for t in TASK:
        b = root / "splits" / t / "build.json"
        if not b.exists():
            continue
        n = {s: v["n"] for s, v in json.loads(b.read_text())["splits"].items()}
        sub = root / "runs" / "subsets" / f"{t}-3k.json"
        if sub.exists():
            d = json.loads(sub.read_text())
            n.update(subset=d["n_records"], subset_groups=d["n_groups"])
        elif t == "i2":
            n["subset"] = n["train"]                    # I2's 3,000-record training split is its own 3k subset
        out["sizes"][t] = n
    if curves:
        i = curves["budgets"].index(0.01)
        out["automation_at_1pct"] = {r: {t: c[i] for t, c in tasks.items()} for r, tasks in curves["rows"].items()
                                     if r in ("kev4b-3k", "sft4b-3k")}
    return out


def load_task5(root: Path, latency: dict, frozen: dict, overlap: dict) -> dict:
    """Sections: median-speedup range per frontier model (non-thinking zero-shot, the H1 cells), training
    context per task (fine-tuned Kev-4B 3k entries of configs/frozen.json), shared patients across splits."""
    out = {"speedup_range": {}, "sft_ratio": {}, "context": {}, "overlap": overlap or {}}
    for model in ("deepseek-v4.1-flash", "kimi-k3"):
        v = [x for c, x in latency.get("speedup_p50", {}).items() if c.startswith(f"{model}/nothink-k0/")]
        if v:
            out["speedup_range"][MODEL[model.replace(".", "")]] = {"min": min(v), "max": max(v)}
    r = list(latency.get("speedup_sft_p50", {}).values())   # LoRA-SFT over Kev-4B per decision (2026-10-26 measurement)
    if r:
        out["sft_ratio"] = {"min": min(r), "max": max(r)}
    for k, e in frozen.get("entries", {}).items():
        row, t, _ = k.split("/")
        if row == "kev4b-3k" and e.get("max_state"):
            out["context"][t] = e["max_state"]
    return out


def load_g4_extras(root: Path) -> dict:
    """Sensitivity analyses: the dev-fixed-threshold sensitivity analysis (runs/devfixed-test.json) and frontier parse-failure rates per model and mode (runs/frontier-parse-failures.json)."""
    out = {}
    f = root / "runs" / "devfixed-test.json"
    if f.exists():
        d = json.loads(f.read_text())
        out["devfixed"] = _ci_dicts({"H3": d["H3_devfixed"], "H4b": d["H4b_devfixed"],
                           "per_task": {t: {r: {"coverage": v["coverage"], "realized_error": v["realized_error"]} for r, v in rows.items()}
                                        for t, rows in d["per_task"].items()}})
    f = root / "runs" / "seed-level-test.json"
    if f.exists():   # M4: seed-level and patient-clustered inference for the H3/H4b secondary metrics
        d = json.loads(f.read_text())
        out["seedlevel"] = _ci_dicts({h: {m: v for m, v in d["two_level"][h].items()} for h in d["two_level"]})
        out["seedlevel_same_sign"] = {m: int(v["same_sign_all_seeds"]) for m, v in d["per_seed"].items()}
        out["seed_task_mean"] = {m: {f"seed{i}": v["task_mean_per_seed"][i] for i in range(3)} for m, v in d["per_seed"].items()}
        out["seed_internal_mean"] = {m: {f"seed{i}": v["internal_mean_per_seed"][i] for i in range(3)} for m, v in d["per_seed"].items()}
    f = root / "runs" / "sensitivity-extra.json"
    if f.exists():   # definitions arm (M1) and I3 without code-bearing reports
        d = json.loads(f.read_text())
        out["defs"] = {t: {c.replace(".", ""): v for c, v in cell.items()} for t, cell in d["definitions"].items()}
        out["defs_think"] = {t: {c.replace(".", ""): v for c, v in cell.items()} for t, cell in d.get("definitions_think", {}).items()}
        out["defs_range"] = {m: v for m, v in d.get("definitions_range", {}).items() if v}
        out["i3nocode"] = {k: v for k, v in d["i3_nocode"].items()}
    f = root / "runs" / "frontier-parse-failures.json"
    if f.exists():
        agg: dict[str, dict] = {}
        for cell, v in json.loads(f.read_text())["runs"].items():
            model, mode, _ = cell.split("/")
            a = agg.setdefault(f"{model.replace('.', '')}/{mode}", {"parse_failures": 0, "questions": 0})
            a["parse_failures"] += v["parse_failures"]; a["questions"] += v["questions"]
        out["parse_failures"] = {k: {**v, "rate": v["parse_failures"] / max(1, v["questions"])} for k, v in agg.items()}
        d = json.loads(f.read_text())["runs"]
        out["gateway"] = {"calls": sum(v["calls"] for v in d.values()), "questions": sum(v["questions"] for v in d.values())}
        out["gateway"]["calls_million"] = out["gateway"]["calls"] / 1e6
        out["gateway"]["questions_million"] = out["gateway"]["questions"] / 1e6
        questions = {cell: v["questions"] for cell, v in d.items()}
    else:
        questions = {}
    f = root / "runs" / "thinking-budget.json"
    if f.exists():   # round 2: thinking traces that reached the 4,096-token budget (retried at 8,192) per model, mode and task
        tb: dict[str, dict] = {}
        for name, v in json.loads(f.read_text()).items():
            model, rest = name.split("-think-")
            k, t = rest.split("-")
            cell = f"{model}/think-{k}/{t}"
            q = questions.get(cell) or v["calls"]
            tb[cell.replace(".", "")] = {"hit_4096": v["ge4096"], "exhausted_8192": v["ge8192"], "questions": q,
                                         "rate_4096": v["ge4096"] / q, "rate_8192": v["ge8192"] / q}
        tot: dict[str, dict] = {}
        for cell, v in tb.items():
            model, mode, _ = cell.split("/")
            a = tot.setdefault(f"{model}/{mode}", {"hit_4096": 0, "questions": 0})
            a["hit_4096"] += v["hit_4096"]; a["questions"] += v["questions"]
        out["thinking_budget"] = {**tb, **{k: {**v, "rate_4096": v["hit_4096"] / v["questions"]} for k, v in tot.items()}}
    f = root / "runs" / "latency.json"
    if f.exists():   # under-load over quiet p50 ratio of the non-thinking zero-shot frontier cells (consistency audit round 2, #5)
        d = json.loads(f.read_text())
        ratios = {c: d["frontier_under_load"][c]["p50_ms"] / d["frontier"][c]["p50_ms"]
                  for c in d.get("frontier_under_load", {}) if c in d.get("frontier", {}) and "nothink-k0" in c}
        if ratios:
            out["load_ratio"] = {"min": min(ratios.values()), "max": max(ratios.values())}
    steps = {}
    for row, name in (("kev4b", "KevThreeK"), ("sft4b", "SftThreeK")):
        f = root / "runs" / f"{row}-p1-3k-s0" / "training_metrics.json"
        if f.exists():
            steps[name] = json.loads(f.read_text())["optimizer_steps"]
    if steps:
        out["steps_p1"] = steps
    f = root / "runs" / "i2-truncation.json"
    if f.exists():
        out["i2_truncation"] = json.loads(f.read_text())["splits"]
    f = root / "runs" / "i3-nocode-rows.json"
    if f.exists():   # round 2: every I3 row on the code-free, unseen-patient test records
        d = json.loads(f.read_text())
        groups: dict[str, list] = {}
        for name, v in d["rows"].items():
            base = name if name.startswith(("deepseek", "kimi")) else name.split("/")[0]   # seeds of one row are averaged
            groups.setdefault(base.replace(".", ""), []).append(v["restricted"])
        rows = {g: {k: sum(x[k] for x in vs) / len(vs) for k in vs[0] if k != "n"} for g, vs in groups.items()}
        out["i3nocode_rows"] = {"rows": rows, "n_records": d["n_records_nocode"], "n_decisions": d["n_decisions_nocode"]}
    return out


def extra_macros(x: dict) -> dict:
    m: dict[str, dict] = {}
    for r, tasks in x.get("test_rows", {}).items():
        for t, v in tasks.items():
            for k in v:
                m[f"{ROW[r]}{TASK[t]}{METRIC[k][0]}"] = {"run": RUN, "key": f"test_rows.{r}.{t}.{k}", "fmt": METRIC[k][1]}
    for cell, v in x.get("frontier_rows", {}).items():
        model, mode, t = cell.split("/")
        for k in v:
            m[f"Front{MODEL[model]}{MODE[mode]}{TASK[t]}{METRIC[k][0]}"] = {"run": RUN, "key": f"frontier_rows.{cell}.{k}", "fmt": METRIC[k][1]}
    for name, tasks in x.get("dev_rows", {}).items():
        for t in tasks:
            m[f"Dev{name}{TASK[t]}Acc"] = {"run": RUN, "key": f"dev_rows.{name}.{t}.accuracy", "fmt": "pct1"}
    for t, n in x.get("sizes", {}).items():
        for k, name in (("train", "Train"), ("dev", "Dev"), ("test", "Test"), ("subset", "Subset"), ("subset_groups", "SubsetGroups")):
            if k in n:
                m[f"N{TASK[t]}{name}"] = {"run": RUN, "key": f"sizes.{t}.{k}", "fmt": ","}
    for model, v in x.get("speedup_range", {}).items():
        for k in ("min", "max"):
            m[f"LatSpeedup{model}{k.capitalize()}"] = {"run": RUN, "key": f"speedup_range.{model}.{k}", "fmt": ".1f" if v[k] < 10 else ".0f"}
    for k, v in x.get("sft_ratio", {}).items():
        m[f"LatSftRatio{k.capitalize()}"] = {"run": RUN, "key": f"sft_ratio.{k}", "fmt": ".1f"}
    for t in x.get("context", {}):
        m[f"Ctx{TASK[t]}"] = {"run": RUN, "key": f"context.{t}", "fmt": ","}
    dv = x.get("devfixed", {})
    for h, name in (("H3", "HThree"), ("H4b", "HFourb")):
        if h in dv:
            _ci(m, f"DevFix{name}Diff", f"devfixed.{h}", "pp1")
    for t, rows in dv.get("per_task", {}).items():
        for r, v in rows.items():
            m[f"DevFix{ROW[r]}{TASK[t]}Cov"] = {"run": RUN, "key": f"devfixed.per_task.{t}.{r}.coverage", "fmt": "pct1"}
            if v.get("realized_error") is not None:
                m[f"DevFix{ROW[r]}{TASK[t]}Err"] = {"run": RUN, "key": f"devfixed.per_task.{t}.{r}.realized_error", "fmt": "pct1"}
    SEEDM = {"automation_at_5pct": ("Auto", "pp1"), "ece": ("Ece", "pp2"), "brier": ("Brier", ".4f"), "accuracy": ("Acc", "pp1"), "macro_f1": ("MacroF", "pp1")}
    for h, metrics in x.get("seedlevel", {}).items():
        name = "HThree" if h == "H3" else "HFourb"
        for met, (mn, fmt) in SEEDM.items():
            if met in metrics:
                _ci(m, f"Seed{name}{mn}Diff", f"seedlevel.{h}.{met}", fmt)
    SEEDN = {"seed0": "Zero", "seed1": "One", "seed2": "Two"}
    for scope, name in (("seed_task_mean", "HThree"), ("seed_internal_mean", "HFourb")):
        for met, vals in x.get(scope, {}).items():
            if met in SEEDM:
                for sd in vals:
                    m[f"Seed{name}{SEEDM[met][0]}Diff{SEEDN[sd]}"] = {"run": RUN, "key": f"{scope}.{met}.{sd}", "fmt": SEEDM[met][1]}
    for t, cell in x.get("defs", {}).items():
        for c, v in cell.items():
            if "/k" in c:
                model, k = c.split("/k")
                m[f"Defs{MODEL[model]}{'Zero' if k == '0' else 'Three'}{TASK[t]}Acc"] = {"run": RUN, "key": f"defs.{t}.{c}.defs.accuracy", "fmt": "pct1"}
                m[f"Defs{MODEL[model]}{'Zero' if k == '0' else 'Three'}{TASK[t]}Gain"] = {"run": RUN, "key": f"defs.{t}.{c}.accuracy_gain", "fmt": "pp1"}
                if k == "0":
                    m[f"DefsPlain{MODEL[model]}{TASK[t]}Acc"] = {"run": RUN, "key": f"defs.{t}.{c}.plain_k0.accuracy", "fmt": "pct1"}
                if "matched_gain" in v:
                    kk = 'Zero' if k == '0' else 'Three'
                    m[f"Defs{MODEL[model]}{kk}{TASK[t]}MatchedGain"] = {"run": RUN, "key": f"defs.{t}.{c}.matched_gain", "fmt": "pp1"}
                    m[f"Defs{MODEL[model]}{kk}{TASK[t]}MatchedGainMacroF"] = {"run": RUN, "key": f"defs.{t}.{c}.matched_gain_macro_f1", "fmt": ".3f"}
                    m[f"DefsPlain{MODEL[model]}{kk}{TASK[t]}Acc"] = {"run": RUN, "key": f"defs.{t}.{c}.plain_same_k.accuracy", "fmt": "pct1"}
                    m[f"DefsPlain{MODEL[model]}{kk}{TASK[t]}MacroF"] = {"run": RUN, "key": f"defs.{t}.{c}.plain_same_k.macro_f1", "fmt": ".3f"}
                    m[f"Defs{MODEL[model]}{kk}{TASK[t]}MacroF"] = {"run": RUN, "key": f"defs.{t}.{c}.defs.macro_f1", "fmt": ".3f"}
            elif c == "kev4b-3k/s0":
                m[f"DefsKev{TASK[t]}Acc"] = {"run": RUN, "key": f"defs.{t}.{c}.accuracy", "fmt": "pct1"}
                m[f"DefsKev{TASK[t]}MacroF"] = {"run": RUN, "key": f"defs.{t}.{c}.macro_f1", "fmt": ".3f"}
    for t, cell in x.get("defs_think", {}).items():   # 2026-10-27: the definitions arm with thinking, matched shot count
        for c, v in cell.items():
            model, k = c.split("/k")
            kk = 'Zero' if k == '0' else 'Three'
            for arm, name in (("defs", "DefsThink"), ("plain_same_k", "DefsThinkPlain")):
                m[f"{name}{MODEL[model]}{kk}{TASK[t]}Acc"] = {"run": RUN, "key": f"defs_think.{t}.{c}.{arm}.accuracy", "fmt": "pct1"}
                m[f"{name}{MODEL[model]}{kk}{TASK[t]}MacroF"] = {"run": RUN, "key": f"defs_think.{t}.{c}.{arm}.macro_f1", "fmt": ".3f"}
            m[f"DefsThink{MODEL[model]}{kk}{TASK[t]}MatchedGain"] = {"run": RUN, "key": f"defs_think.{t}.{c}.matched_gain", "fmt": "pp1"}
            m[f"DefsThink{MODEL[model]}{kk}{TASK[t]}MatchedGainMacroF"] = {"run": RUN, "key": f"defs_think.{t}.{c}.matched_gain_macro_f1", "fmt": ".3f"}
    for mode, v in x.get("defs_range", {}).items():
        name = {"nothink": "Defs", "think": "DefsThink"}[mode]
        for b in ("min", "max"):
            m[f"{name}MatchedGain{b.title()}"] = {"run": RUN, "key": f"defs_range.{mode}.{b}", "fmt": "pp1"}
    for row, v in x.get("i3nocode", {}).items():
        if row == "counts":
            for split, c in v.items():
                for k2 in ("n", "code_bearing", "kept"):
                    m[f"IThreeNoCode{split.capitalize()}{k2.title().replace('_', '')}"] = {"run": RUN, "key": f"i3nocode.counts.{split}.{k2}", "fmt": ","}
            continue
        name = {"kev4b": "KevThreeK", "sft4b": "SftThreeK"}[row]
        m[f"IThreeNoCode{name}Acc"] = {"run": RUN, "key": f"i3nocode.{row}.retrained_nocode.accuracy", "fmt": "pct1"}
        m[f"IThreeNoCode{name}OrigAcc"] = {"run": RUN, "key": f"i3nocode.{row}.original_3k_same_records.accuracy", "fmt": "pct1"}
        m[f"IThreeNoCode{name}Auto"] = {"run": RUN, "key": f"i3nocode.{row}.retrained_nocode.automation_at_5pct", "fmt": "pct1"}
    for cell, v in x.get("parse_failures", {}).items():
        model, mode = cell.split("/")
        m[f"ParseFail{MODEL[model]}{MODE[mode]}Rate"] = {"run": RUN, "key": f"parse_failures.{cell}.rate", "fmt": "pct2"}
        m[f"ParseFail{MODEL[model]}{MODE[mode]}N"] = {"run": RUN, "key": f"parse_failures.{cell}.parse_failures", "fmt": ","}
    for model, modes in x.get("h2_fixed_mode", {}).items():
        for mode in modes:
            m[f"HTwo{MODEL[model]}Closure{MODE[mode]}"] = {"run": RUN, "key": f"h2_fixed_mode.{model}.{mode}", "fmt": ".2f"}
    if "gateway" in x:
        m["GatewayCalls"] = {"run": RUN, "key": "gateway.calls", "fmt": ","}
        m["GatewayQuestions"] = {"run": RUN, "key": "gateway.questions", "fmt": ","}
        m["GatewayCallsMillion"] = {"run": RUN, "key": "gateway.calls_million", "fmt": ".2f"}
        m["GatewayQuestionsMillion"] = {"run": RUN, "key": "gateway.questions_million", "fmt": ".2f"}
    for cell, v in x.get("thinking_budget", {}).items():
        parts = cell.split("/")
        model, mode = parts[0], parts[1]
        suffix = TASK[parts[2]] if len(parts) == 3 else ""
        m[f"ThinkBudget{MODEL[model]}{MODE[mode]}{suffix}Rate"] = {"run": RUN, "key": f"thinking_budget.{cell}.rate_4096", "fmt": "pct1"}
        if len(parts) == 3:
            m[f"ThinkBudget{MODEL[model]}{MODE[mode]}{suffix}ExhaustedRate"] = {"run": RUN, "key": f"thinking_budget.{cell}.rate_8192", "fmt": "pct2"}
    for k in x.get("load_ratio", {}):
        m[f"LatLoadRatio{k.capitalize()}"] = {"run": RUN, "key": f"load_ratio.{k}", "fmt": ".0f"}
    for name in x.get("steps_p1", {}):
        m[f"Steps{name}POne"] = {"run": RUN, "key": f"steps_p1.{name}", "fmt": ","}
    for split, v in x.get("i2_truncation", {}).items():
        m[f"ITwo{split.capitalize()}NoHeadingShare"] = {"run": RUN, "key": f"i2_truncation.{split}.share", "fmt": "pct1"}
        m[f"ITwo{split.capitalize()}NoHeadingN"] = {"run": RUN, "key": f"i2_truncation.{split}.no_heading", "fmt": ","}
    nc = x.get("i3nocode_rows", {})
    if nc:
        m["IThreeNoCodeRecords"] = {"run": RUN, "key": "i3nocode_rows.n_records", "fmt": ","}
        m["IThreeNoCodeDecisions"] = {"run": RUN, "key": "i3nocode_rows.n_decisions", "fmt": ","}
    for g, v in nc.get("rows", {}).items():
        if "/" in g:
            model, mode = g.split("/")
            name = f"Front{MODEL[model]}{MODE[mode]}"
        else:
            name = ROW.get(g.replace("-nocode", ""), g) + ("Retrained" if g.endswith("-nocode") else "")
        for k, (mn, fmt) in METRIC.items():
            if k in v:
                m[f"IThreeNoCodeRow{name}{mn}"] = {"run": RUN, "key": f"i3nocode_rows.rows.{g}.{k}", "fmt": fmt}
    for t, v in x.get("overlap", {}).items():
        for k in v:
            m[f"Overlap{TASK[t]}" + "".join(w.capitalize() for w in k.split("_"))] = {"run": RUN, "key": f"overlap.{t}.{k}", "fmt": ","}
    for t, v in x.get("majority", {}).items():
        m[f"Majority{TASK[t]}Acc"] = {"run": RUN, "key": f"majority.{t}.accuracy", "fmt": "pct1"}
        m[f"NRows{TASK[t]}"] = {"run": RUN, "key": f"majority.{t}.n_rows", "fmt": ","}
        if "macro_f1" in v:
            m[f"Majority{TASK[t]}MacroF"] = {"run": RUN, "key": f"majority.{t}.macro_f1", "fmt": ".3f"}
    for split, v in x.get("i3_codes", {}).items():
        name = split.capitalize()
        m[f"IThreeCode{name}"] = {"run": RUN, "key": f"i3_codes.{split}.with_code", "fmt": ","}
        m[f"IThreeCode{name}Share"] = {"run": RUN, "key": f"i3_codes.{split}.share", "fmt": "pct1"}
        m[f"IThreeCode{name}Agree"] = {"run": RUN, "key": f"i3_codes.{split}.agree_share", "fmt": "pct1"}
    for t, sides in x.get("seed_spread", {}).items():
        for side, metrics in sides.items():
            for k, (name, fmt) in {"automation_at_5pct": ("Auto", "pp1"), "ece": ("Ece", "pp2")}.items():
                if k in metrics:
                    for stat in ("min", "max", "sd"):
                        m[f"Seed{side.capitalize()}{TASK[t]}{name}{stat.capitalize()}"] = {"run": RUN, "key": f"seed_spread.{t}.{side}.{k}.{stat}", "fmt": fmt}
    for r, tasks in x.get("automation_at_1pct", {}).items():
        for t in tasks:
            m[f"{ROW[r]}{TASK[t]}AutoOne"] = {"run": RUN, "key": f"automation_at_1pct.{r}.{t}", "fmt": "pct1"}
    return m


AGE = {"0-17": "AgeZero", "18-39": "AgeEighteen", "40-59": "AgeForty", "60-79": "AgeSixty", "80+": "AgeEighty", "missing": "AgeMissing"}
SEX = {"female": "Female", "male": "Male", "other or unknown": "SexOther"}
SUBROW = {"kev": "KevThreeK", "sft": "SftThreeK", "frontier": "Front"}


def load_g4_optional(root: Path) -> dict:
    """Post hoc analyses: per-task paired differences against the
    frontier (runs/pertask-frontier.json), H2/H3/H4b with I3 restricted to the code-free records
    (runs/hypotheses-i3nocode-test.json), subgroups by age band and sex (runs/subgroups.json)."""
    out: dict = {}
    f = root / "runs" / "pertask-frontier.json"
    if f.exists():
        d = json.loads(f.read_text())["tasks"]
        out["pertask"] = _ci_dicts(d)
        counts = {}
        for row in ("kev", "sft"):
            for met in ("macro_f1", "accuracy"):
                k = f"{row}_minus_fr_{met}"
                above = [t for t, v in d.items() if all(v[mo][k]["ci95"][0] > 0 for mo in v)]
                below = [t for t, v in d.items() if all(v[mo][k]["ci95"][1] < 0 for mo in v)]
                counts[f"{row}_{met}"] = {"above_both": len(above), "below_both": len(below)}
        out["pertask_counts"] = counts
        out["pertask_dropped"] = {t: max(v[mo].get("dropped_pairs", 0) for mo in v) for t, v in d.items()}
    f = root / "runs" / "hypotheses-i3nocode-test.json"
    if f.exists():
        d = add_two_sided(json.loads(f.read_text()))
        out["i3nocode_hyp"] = _ci_dicts({k: d[k] for k in ("H2", "H3", "H4b") if d.get(k, {}).get("status") == "ok"})
    f = root / "runs" / "subgroups.json"
    if f.exists():
        out["subgroups"] = json.loads(f.read_text())["tasks"]
    return out


def optional_macros(x: dict) -> dict:
    m: dict[str, dict] = {}
    for t, models in x.get("pertask", {}).items():
        for model, v in models.items():
            for row, rn in (("kev", "Kev"), ("sft", "Sft")):
                for met, (mn, fmt) in (("accuracy", ("Acc", "pp1")), ("macro_f1", ("MacroF", ".3f"))):
                    key = f"pertask.{t}.{model}.{row}_minus_fr_{met}"
                    _ci(m, f"PerTask{rn}Minus{MODEL[model]}{TASK[t]}{mn}", key, fmt)
                    m[f"PerTask{rn}Minus{MODEL[model]}{TASK[t]}{mn}P"] = {"run": RUN, "key": f"{key}.p_two_sided", "fmt": ".3f"}
    for k, v in x.get("pertask_counts", {}).items():
        row, met = k.split("_", 1)
        for side in v:
            m[f"PerTask{row.capitalize()}{METRIC[met][0]}{''.join(w.capitalize() for w in side.split('_'))}"] = {"run": RUN, "key": f"pertask_counts.{k}.{side}", "fmt": "d"}
    for t in x.get("pertask_dropped", {}):
        m[f"PerTaskDropped{TASK[t]}"] = {"run": RUN, "key": f"pertask_dropped.{t}", "fmt": ","}
    h = x.get("i3nocode_hyp", {})
    for hyp, prefix in (("H3", "HThree"), ("H4b", "HFourb")):
        if hyp not in h:
            continue
        base = f"i3nocode_hyp.{hyp}"
        _ci(m, f"{prefix}NoCodeDiff", f"{base}.primary_mean_difference", "pp1")
        for sec, name in (("accuracy", "Acc"), ("macro_f1", "MacroF"), ("ece", "Ece"), ("brier", "Brier")):
            _ci(m, f"{prefix}NoCode{name}Diff", f"{base}.secondary.{sec}", {"accuracy": "pp1", "macro_f1": "pp1", "ece": "pp2"}.get(sec, ".4f"))
            m[f"{prefix}NoCode{name}PHolm"] = {"run": RUN, "key": f"{base}.secondary.{sec}.p_holm", "fmt": ".3f"}
            m[f"{prefix}NoCode{name}PTwoSided"] = {"run": RUN, "key": f"{base}.secondary.{sec}.p_two_sided", "fmt": ".3f"}
    for model in h.get("H2", {}).get("models", {}):
        _ci(m, f"HTwo{MODEL[model]}ClosureNoCode", f"i3nocode_hyp.H2.models.{model}.closure", ".2f")
    for t, r in x.get("subgroups", {}).items():
        for dim, names in (("age", AGE), ("sex", SEX), ("all", None)):
            groups = {"all": r["all"]} if dim == "all" else r[dim]
            for g, rows in groups.items():
                gn = "All" if dim == "all" else names[g]
                for row, c in rows.items():
                    base = f"subgroups.{t}.{dim}" + ("" if dim == "all" else f".{g}") + f".{row}"
                    if row == "kev":
                        m[f"Sub{TASK[t]}{gn}N"] = {"run": RUN, "key": f"{base}.n", "fmt": ","}
                    if not c.get("suppressed"):
                        m[f"Sub{TASK[t]}{gn}{SUBROW[row]}Acc"] = {"run": RUN, "key": f"{base}.accuracy", "fmt": "pct1"}
                        m[f"Sub{TASK[t]}{gn}{SUBROW[row]}MacroF"] = {"run": RUN, "key": f"{base}.macro_f1", "fmt": ".3f"}
    return m


def add_two_sided(h: dict) -> dict:
    """Two-sided bootstrap P for each secondary metric from the one-sided bootstrap share (2 x min(p, 1 - p)); the
    one-sided Holm-adjusted P stays the prespecified quantity."""
    for hyp in ("H3", "H4b"):
        for sec in h.get(hyp, {}).get("secondary", {}).values():
            pk = sec.get("p_kev_better")
            if pk is not None:
                sec["p_two_sided"] = min(1.0, 2 * min(pk, 1 - pk))
    return h


def macros_for(h: dict, lat: dict) -> dict:
    m: dict[str, dict] = {}
    for hyp, prefix in (("H3", "HThree"), ("H4b", "HFourb")):
        r = h.get(hyp, {})
        if r.get("status") != "ok":
            continue
        base = f"hypotheses.{hyp}"
        _ci(m, f"{prefix}Diff", f"{base}.primary_mean_difference", "pp1")
        m[f"{prefix}P"] = {"run": RUN, "key": f"{base}.primary_mean_difference.p_one_sided_gt0", "fmt": ".3f"}
        for sec, name in (("accuracy", "Acc"), ("macro_f1", "MacroF"), ("ece", "Ece"), ("brier", "Brier")):
            if sec not in r["secondary"]:
                continue
            _ci(m, f"{prefix}{name}Diff", f"{base}.secondary.{sec}", {"accuracy": "pp1", "macro_f1": "pp1", "ece": "pp2"}.get(sec, ".4f"))
            m[f"{prefix}{name}PHolm"] = {"run": RUN, "key": f"{base}.secondary.{sec}.p_holm", "fmt": ".3f"}
            if "p_two_sided" in r["secondary"][sec]:
                m[f"{prefix}{name}PTwoSided"] = {"run": RUN, "key": f"{base}.secondary.{sec}.p_two_sided", "fmt": ".3f"}
        for t in r["tasks"]:
            for side in ("kev", "sft"):
                m[f"{prefix}{TASK[t]}{side.capitalize()}"] = {"run": RUN, "key": f"{base}.per_task.{t}.automation_at_5pct.{side}", "fmt": "pct1"}
                m[f"{prefix}{TASK[t]}{side.capitalize()}Acc"] = {"run": RUN, "key": f"{base}.per_task.{t}.accuracy.{side}", "fmt": "pct1"}
                if "ece" in r["per_task"][t]:
                    m[f"{prefix}{TASK[t]}{side.capitalize()}Ece"] = {"run": RUN, "key": f"{base}.per_task.{t}.ece.{side}", "fmt": "pct2"}
                    m[f"{prefix}{TASK[t]}{side.capitalize()}Brier"] = {"run": RUN, "key": f"{base}.per_task.{t}.brier.{side}", "fmt": ".4f"}
            _ci(m, f"{prefix}{TASK[t]}Diff", f"{base}.per_task_primary.{t}", "pp1")
        for t, n in r.get("dropped_pairs", {}).items():
            m[f"{prefix}{TASK[t]}Dropped"] = {"run": RUN, "key": f"{base}.dropped_pairs.{t}", "fmt": "d"}
    r = h.get("H1", {})
    if r.get("status") == "ok":
        _ci(m, "HOneRatio", "hypotheses.H1.ratio_kev_over_qwen", ".3f")
        for t, v in r["per_task_macro_f1"].items():
            for side in v:
                m[f"HOne{TASK[t]}{side.capitalize()}"] = {"run": RUN, "key": f"hypotheses.H1.per_task_macro_f1.{t}.{side}", "fmt": ".3f"}
    r = h.get("H2", {})
    if r.get("status") == "ok":
        for model, v in r["models"].items():
            _ci(m, f"HTwo{MODEL[model]}Closure", f"hypotheses.H2.models.{model}.closure", ".2f")
            # descriptive sensitivity: point estimate without P1 (labeler) and I3 (codes in text)
            m[f"HTwo{MODEL[model]}ClosureNoPOneIThree"] = {"run": RUN, "key": f"h2_sensitivity.{model}.without_p1_i3", "fmt": ".2f"}
            for t in v["per_task"]:
                for f, name in (("zero_shot", "Zero"), ("fine_tuned", "Tuned"), ("frontier", "Frontier")):
                    m[f"HTwo{MODEL[model]}{TASK[t]}{name}"] = {"run": RUN, "key": f"hypotheses.H2.models.{model}.per_task.{t}.{f}", "fmt": ".3f"}
    for t, v in lat.get("kev", {}).items():
        m[f"LatKev{TASK[t]}"] = {"run": RUN, "key": f"latency.kev.{t}.p50_ms", "fmt": ",.0f"}
        m[f"LatKevPNinetyFive{TASK[t]}"] = {"run": RUN, "key": f"latency.kev.{t}.p95_ms", "fmt": ",.0f"}
    for cell in lat.get("frontier", {}):
        model, mode, t = cell.split("/")
        m[f"LatFront{MODEL[model]}{MODE[mode]}{TASK[t]}"] = {"run": RUN, "key": f"latency.frontier.{cell}.p50_ms", "fmt": ",.0f"}
    for cell, v in lat.get("speedup_p50", {}).items():
        model, mode, t = cell.split("/")
        m[f"LatSpeedup{MODEL[model]}{MODE[mode]}{TASK[t]}"] = {"run": RUN, "key": f"latency.speedup_p50.{cell}", "fmt": ".1f" if v < 10 else ".0f"}
    # 2026-10-26: LoRA-SFT under Kev's conditions, and the test-run (under own load) frontier latencies kept for the record
    for t, v in lat.get("sft", {}).items():
        m[f"LatSft{TASK[t]}"] = {"run": RUN, "key": f"latency.sft.{t}.p50_ms", "fmt": ",.0f"}
        m[f"LatSftPNinetyFive{TASK[t]}"] = {"run": RUN, "key": f"latency.sft.{t}.p95_ms", "fmt": ",.0f"}
    for t in lat.get("speedup_sft_p50", {}):
        m[f"LatSpeedupSft{TASK[t]}"] = {"run": RUN, "key": f"latency.speedup_sft_p50.{t}", "fmt": ".1f"}
    for cell in lat.get("frontier_under_load", {}):
        model, mode, t = cell.split("/")
        m[f"LatFrontLoad{MODEL[model]}{MODE[mode]}{TASK[t]}"] = {"run": RUN, "key": f"latency.frontier_under_load.{cell}.p50_ms", "fmt": ",.0f"}
    return m


def safe_cells(latency: dict) -> dict:
    """Build_results splits keys on '.', so the frontier cell names lose their dots (deepseek-v4.1-flash -> deepseek-v41-flash)."""
    return {k: ({c.replace(".", ""): v for c, v in latency[k].items()} if k in ("frontier", "speedup_p50", "frontier_under_load", "speedup_under_load_p50") else latency[k])
            for k in latency}


def h2_sensitivity(h: dict) -> dict:
    """Gap closure recomputed from H2's own paired per-task values without P1 and I3; point estimates only."""
    out = {}
    for model, v in h.get("H2", {}).get("models", {}).items():
        if v.get("status") != "ok":
            continue
        ts = [t for t in v["per_task"] if t not in ("p1", "i3")]
        num = sum(v["per_task"][t]["fine_tuned"] - v["per_task"][t]["zero_shot"] for t in ts)
        den = sum(v["per_task"][t]["frontier"] - v["per_task"][t]["zero_shot"] for t in ts)
        out[model] = {"without_p1_i3": num / den}
    return out


def h2_fixed_mode(rows: dict, frontier: dict) -> dict:
    """Gap closure with one fixed non-thinking frontier mode for every task instead of the best of four:
    the same means over the six tasks, from the unpaired test summaries (point estimates only)."""
    out: dict = {}
    tasks = ("p1", "p2", "p3", "i1", "i2", "i3")
    try:
        tuned = {t: rows["kev4b-3k"][t][1] for t in tasks}
        zero = {t: rows["kev4b-zeroshot"][t][1] for t in tasks}
    except (KeyError, TypeError):
        return out
    gain = sum(tuned[t] - zero[t] for t in tasks) / len(tasks)
    for model in ("deepseek-v4.1-flash", "kimi-k3"):
        for mode in ("nothink-k0", "nothink-k3", "think-k0", "think-k3"):
            cells = [frontier.get(f"{model}/{mode}/{t}") for t in tasks]
            if any(c is None or c[1] is None for c in cells):
                continue
            lead = sum(c[1] - zero[t] for c, t in zip(cells, tasks)) / len(tasks)
            out.setdefault(model.replace(".", ""), {})[mode] = gain / lead
    return out


def write(root: Path, hypotheses: dict, latency: dict, extra: dict | None = None) -> None:
    root = Path(root)
    lat = safe_cells(latency)
    metrics = {"hypotheses": _ci_dicts(hypotheses), "latency": _ci_dicts(lat), "h2_sensitivity": h2_sensitivity(hypotheses), **(extra or {})}
    (root / "runs" / RUN).mkdir(parents=True, exist_ok=True)
    (root / "runs" / RUN / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hypotheses", default="runs/hypotheses-test.json")
    ap.add_argument("--latency", default="runs/latency.json")
    ap.add_argument("--test_rows", default="runs/test-rows-summary.json")
    ap.add_argument("--frontier", default="runs/frontier-test-summary.json")
    ap.add_argument("--curves", default="runs/curves-test.json")
    ap.add_argument("--overlap", default="runs/group-overlap.json")
    ap.add_argument("--root", default=".")
    a = ap.parse_args(argv)
    root = Path(a.root)
    read = lambda f: json.loads((root / f).read_text()) if (root / f).exists() else None
    extra = load_extra(root, read(a.test_rows) or {}, read(a.frontier) or {}, read(a.curves))
    extra.update(load_task5(root, read(a.latency) or {}, read("configs/frozen.json") or {}, read(a.overlap) or {}))
    extra.update(load_g4_extras(root))
    extra.update(load_g4_optional(root))
    extra["h2_fixed_mode"] = h2_fixed_mode(read(a.test_rows) or {}, read(a.frontier) or {})
    extra["majority"] = read("runs/majority-test.json") or {}
    extra["i3_codes"] = {k: {**v, "share": v["with_code"] / v["n_records"], "agree_share": v["label_agrees"] / max(v["with_code"], 1)}
                         for k, v in (read("runs/i3-codes.json") or {}).items()}
    extra["seed_spread"] = {t: {side: {k: {s: v[s] for s in ("min", "max", "sd")} for k, v in ms.items()} for side, ms in sides.items()}
                            for t, sides in (read("runs/seed-spread-test.json") or {}).items()}
    write(root, add_two_sided(read(a.hypotheses)), read(a.latency) or {}, extra)
    print(f"wrote runs/{RUN}/metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
