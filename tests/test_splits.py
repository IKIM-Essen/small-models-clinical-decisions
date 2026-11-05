import json
from collections import Counter

import pytest

from kevmed import schema, splits


def _recs(n, groups_per=2):
    out = []
    for i in range(n):
        g = f"g{i // groups_per}"
        out.append(schema.record(f"r{i}", "s", {"a": schema.choice("?", {"x": None, "y": None}, "x" if i % 3 else "y")},
                                 {"group": g}))
    return out


def test_group_split_keeps_groups_together_and_is_seeded():
    recs = _recs(300)
    groups = {r["_meta"]["id"]: r["_meta"]["group"] for r in recs}
    labels = {r["_meta"]["id"]: r["questions"]["a"]["label"] for r in recs}
    fr = {"train": 0.7, "dev": 0.1, "test": 0.2}
    a = splits.group_split(groups, labels, fr, seed=1)
    b = splits.group_split(groups, labels, fr, seed=1)
    c = splits.group_split(groups, labels, fr, seed=2)
    assert a == b and a != c
    by_group = {}
    for rid, sp in a.items():
        by_group.setdefault(groups[rid], set()).add(sp)
    assert all(len(s) == 1 for s in by_group.values())
    counts = Counter(a.values())
    assert 180 <= counts["train"] <= 240 and 40 <= counts["test"] <= 80


def test_group_split_stratifies_by_label():
    recs = _recs(600, groups_per=1)
    groups = {r["_meta"]["id"]: r["_meta"]["group"] for r in recs}
    labels = {r["_meta"]["id"]: r["questions"]["a"]["label"] for r in recs}
    sp = splits.group_split(groups, labels, {"train": 0.8, "test": 0.2}, seed=3)
    share = lambda s: sum(labels[r] == "y" for r, v in sp.items() if v == s) / sum(v == s for v in sp.values())
    assert abs(share("train") - share("test")) < 0.06


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        splits.group_split({"a": "a"}, {"a": "x"}, {"train": 0.5, "test": 0.4}, seed=0)


def test_write_splits_writes_jsonl_and_build_json_and_refuses_overwrite(tmp_path):
    recs = _recs(10)
    by = {"train": recs[:7], "dev": recs[7:8], "test": recs[8:]}
    info = splits.write_splits(tmp_path / "s9", by, {"task": "s9", "seed": 1})
    assert (tmp_path / "s9" / "train.jsonl").exists()
    b = json.loads((tmp_path / "s9" / "build.json").read_text())
    assert b["task"] == "s9" and b["splits"]["train"]["n"] == 7 and len(b["files"]["train.jsonl"]["sha256"]) == 64
    assert b["git_commit"] and info == b
    with pytest.raises(FileExistsError):
        splits.write_splits(tmp_path / "s9", by, {"task": "s9", "seed": 1})


def test_overlap_counts_groups_in_more_than_one_split():
    recs = _recs(4, groups_per=2)
    assert splits.overlap({"train": recs[:1], "test": recs[1:]}, "group") == 1
    assert splits.overlap({"train": recs[:2], "test": recs[2:]}, "group") == 0


def test_restricted_root_defaults_to_the_repo_checkout(monkeypatch):
    """Rows live in the repo checkout (git-ignored), so the guards need no env var."""
    from kevmed import splits
    monkeypatch.delenv("KEV_RESTRICTED_ROOT", raising=False)
    assert splits.DEFAULT_RESTRICTED_ROOT == str(splits.REPO_ROOT)
