import importlib.util, json
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "subset_train.py"

@pytest.fixture(scope="module")
def st():
    spec = importlib.util.spec_from_file_location("subset_train", SCRIPT)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def recs(groups):  # groups: list of group sizes
    out = []
    for g, size in enumerate(groups):
        for i in range(size):
            out.append({"_meta": {"id": f"{g}-{i}", "group": f"g{g}"}, "state": "synthetic",
                        "questions": {"q": {"type": "noul", "label": "true" if i % 2 else "false"}}})
    return out

def test_never_more_than_n(st):
    assert len(st.sample_by_group(recs([3] * 50), n=20, seed=1)) <= 20

def test_fills_to_n_when_groups_allow(st):
    assert len(st.sample_by_group(recs([1] * 50), n=20, seed=1)) == 20

def test_groups_are_never_split(st):
    data = recs([2, 3, 4, 5] * 10)
    picked = st.sample_by_group(data, n=17, seed=3)
    sizes = {}
    for r in data: sizes[r["_meta"]["group"]] = sizes.get(r["_meta"]["group"], 0) + 1
    got = {}
    for r in picked: got[r["_meta"]["group"]] = got.get(r["_meta"]["group"], 0) + 1
    assert all(got[g] == sizes[g] for g in got)

def test_same_seed_same_subset(st):
    data = recs([1, 2, 3] * 30)
    ids = lambda rs: [r["_meta"]["id"] for r in rs]
    assert ids(st.sample_by_group(data, 25, 7)) == ids(st.sample_by_group(data, 25, 7))

def test_other_seed_other_subset(st):
    data = recs([1] * 200)
    ids = lambda rs: [r["_meta"]["id"] for r in rs]
    assert ids(st.sample_by_group(data, 25, 7)) != ids(st.sample_by_group(data, 25, 8))

def test_small_split_returned_whole(st):
    data = recs([2] * 5)
    assert len(st.sample_by_group(data, n=3000, seed=1)) == 10

def test_keeps_source_order(st):
    data = recs([1] * 100)
    pos = {r["_meta"]["id"]: i for i, r in enumerate(data)}
    picked = st.sample_by_group(data, 30, 2)
    assert [pos[r["_meta"]["id"]] for r in picked] == sorted(pos[r["_meta"]["id"]] for r in picked)

def test_cli_writes_subset_and_counts_only_summary(st, tmp_path, capsys):
    src = tmp_path / "train.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in recs([1] * 40)))
    out, summ = tmp_path / "sub" / "train.jsonl", tmp_path / "summary.json"
    assert st.main(["--data", str(src), "--out", str(out), "--summary", str(summ), "--n", "10", "--seed", "5"]) == 0
    assert sum(1 for _ in out.open()) == 10
    s = json.loads(summ.read_text())
    assert s["n_records"] == 10 and s["n_requested"] == 10 and s["seed"] == 5
    assert set(s["labels"]["q"]) <= {"true", "false"}
    assert "synthetic" not in summ.read_text() and "synthetic" not in capsys.readouterr().out


def test_cli_refuses_to_overwrite_an_existing_subset(st, tmp_path):
    src = tmp_path / "train.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in recs([1] * 40)))
    out, summ = tmp_path / "train-3k.jsonl", tmp_path / "summary.json"
    args = ["--data", str(src), "--out", str(out), "--summary", str(summ), "--n", "10", "--seed", "5"]
    assert st.main(args) == 0
    before = out.read_text()
    assert st.main(args[:-1] + ["6"]) != 0
    assert out.read_text() == before
