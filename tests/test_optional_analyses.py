"""Post hoc analyses: the fast record bootstrap matches st.bootstrap, reliability bins, subgroup
groupings and the optional result values. Synthetic rows only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
import pertask_frontier as pf  # noqa: E402
import reliability as rel  # noqa: E402
import results_macros as rm  # noqa: E402
import stats as st  # noqa: E402
import subgroups as sg  # noqa: E402


def rows(seed, n=40, q=("a", "b")):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        for qq in q[: 1 + i % len(q)]:
            p = rng.dirichlet([1, 1, 1])
            out.append({"id": f"r{i:03d}", "question": qq, "label": int(rng.integers(0, 3)), "p": p.tolist()})
    return sorted(out, key=st.key)


def test_fast_bootstrap_identical_to_stats_bootstrap():
    tasks = {"x": {"a": [rows(1), rows(2)], "b": [rows(3)]}, "y": {"a": [rows(4, 25, ("a",))], "b": [rows(5, 25, ("a",))]}}
    for t in tasks.values():          # same pairs in every run
        keys = [st.key(r) for r in next(iter(t.values()))[0]]
        for seeds in t.values():
            for rs in seeds:
                assert [st.key(r) for r in rs] == keys
    stat = lambda d: st.mean_over_tasks(d, lambda r: st.seed_mean(r["a"], st.macro_f1) - st.seed_mean(r["b"], st.accuracy))
    assert np.array_equal(st.bootstrap(tasks, stat, 50, 9), pf.fast_bootstrap(tasks, stat, 50, 9))


def test_draw_index_matches_span_concatenation():
    rs = rows(7)
    starts, lens = pf.record_index(rs)
    spans = st._record_spans(rs)
    picks = np.array([3, 0, 3, 11, 39])
    assert np.array_equal(pf.draw_index(starts, lens, picks), np.concatenate([spans[p] for p in picks]))


def test_two_sided():
    assert pf.two_sided(np.array([1.0, 2.0, 3.0, -1.0])) == 0.5
    assert pf.two_sided(np.array([1.0, 2.0])) == 0.0


def test_reliability_bins_and_ece():
    conf = np.array([0.05, 0.15, 0.95, 1.0, 0.55])
    correct = np.array([0, 1, 1, 1, 0], dtype=bool)
    bs = rel.bins(conf, correct)
    assert [b["n"] for b in bs] == [1, 1, 0, 0, 0, 1, 0, 0, 0, 2]
    assert bs[9]["accuracy"] == 1.0 and bs[2]["confidence"] is None
    assert abs(rel.ece_from_bins(bs) - (0.05 + 0.85 + 0.55 + 0.025 * 2) / 5) < 1e-12


def test_subgroup_groupings():
    assert [sg.age_band(a) for a in (0, 17.9, 18, 39, 40, 79.5, 80, 91, None, "x")] == \
        ["0-17", "0-17", "18-39", "18-39", "40-59", "60-79", "80+", "80+", "missing", "missing"]
    assert [sg.sex_group(s) for s in ("F", "male", "m", "Female", None, "other")] == \
        ["female", "male", "male", "female", "other or unknown", "other or unknown"]
    rs = [[{"id": f"r{i}", "question": "q", "label": i % 2, "p": [0.6, 0.4]} for i in range(12)]]
    assert sg.cell(rs, {f"r{i}" for i in range(9)}) == {"n": 9, "suppressed": True}
    c = sg.cell(rs, {f"r{i}" for i in range(12)})
    assert c["n"] == 12 and c["accuracy"] == 0.5


def test_optional_macros_and_tables(tmp_path):
    ci = lambda v: {"point": v, "ci95": [v - 0.01, v + 0.01], "p_two_sided": 0.01}
    model = {"mode": "nothink-k3", **{f"{r}_minus_fr_{m}": ci(0.05) for r in ("kev", "sft") for m in ("accuracy", "macro_f1")}}
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "pertask-frontier.json").write_text(json.dumps({"tasks": {"p2": {"deepseek": model, "kimi": model}}}))
    c = lambda n: {"n": n, "accuracy": 0.5, "macro_f1": 0.4} if n >= 10 else {"n": n, "suppressed": True}
    (tmp_path / "runs" / "subgroups.json").write_text(json.dumps({"tasks": {"p3": {
        "all": {"kev": c(30), "sft": c(30), "frontier": c(30)},
        "age": {"18-39": {"kev": c(25), "sft": c(25), "frontier": c(25)}, "80+": {"kev": c(5), "sft": c(5), "frontier": c(5)}},
        "sex": {"female": {"kev": c(30), "sft": c(30), "frontier": c(30)}}}}}))
    x = rm.load_g4_optional(tmp_path)
    m = rm.optional_macros(x)
    assert x["pertask_counts"]["kev_macro_f1"] == {"above_both": 1, "below_both": 0}
    assert m["PerTaskKevMinusDeepseekPTwoMacroF"]["key"] == "pertask.p2.deepseek.kev_minus_fr_macro_f1.point"
    assert m["PerTaskKevMacroFAboveBoth"]["fmt"] == "d"
    assert "SubPThreeAgeEighteenKevThreeKAcc" in m and "SubPThreeAgeEightyN" in m and "SubPThreeAgeEightyKevThreeKAcc" not in m


