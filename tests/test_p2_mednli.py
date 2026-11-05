import pytest

from conftest import REPO
from kevmed.tasks import p2_mednli as p2

FX = REPO / "tests/fixtures/physionet"


def test_unknown_label_fails_with_pair_id_only():
    with pytest.raises(ValueError) as e:
        p2.build(FX, seed=1)
    assert "train-008" in str(e.value) and "SYNTHETIC" not in str(e.value)


def test_published_splits_and_record_shape():
    by, consort = p2.build(FX, seed=1, skip_bad=True)
    assert (len(by["train"]), len(by["dev"]), len(by["test"])) == (8, 3, 3) and consort["dropped_unknown_label"] == 1
    r = by["dev"][0]
    assert set(r["state"]) == {"premise", "hypothesis"} and set(r["questions"]["relation"]["criteria"]) == {"entailment", "contradiction", "neutral"}
    assert r["_meta"]["id"] == "mednli-dev-000"
