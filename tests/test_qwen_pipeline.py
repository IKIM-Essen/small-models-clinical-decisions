"""Scripts/qwen/: Qwen3.5-4B Instruct predictor and LoRA-SFT baseline. Synthetic records,
a character-level fake tokenizer and a fake model only: no weights, no PhysioNet or internal rows."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

QWEN = Path(__file__).resolve().parents[1] / "scripts" / "qwen"
sys.path.insert(0, str(QWEN))


def load(name):
    spec = importlib.util.spec_from_file_location(name, QWEN / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ql, bq, ts = load("qwen_lib"), load("bench_qwen"), load("train_sft")
from kev.model import ContextOverflow  # noqa: E402


class FakeTok:
    """One token per character, so every option letter is a single token and lengths are easy to reason about."""
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        assert tokenize is False and add_generation_prompt and enable_thinking is False
        return "".join(f"<{m['role']}>{m['content']}" for m in msgs) + "<assistant>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


class FakeCausalLM:
    """Causal by construction: the hidden state at t is the running mean of the token embeddings up to t, so tokens after
    t (right padding) cannot change it. `.model` / `.lm_head` like a *ForCausalLM."""
    def __init__(self, vocab=256, dim=8):
        g = torch.Generator().manual_seed(0)
        self.emb = torch.randn(vocab, dim, generator=g)
        self.lm_head = torch.nn.Linear(dim, vocab, bias=False)
        torch.nn.init.normal_(self.lm_head.weight, generator=g)
        self.calls = 0

    def model(self, input_ids, attention_mask, use_cache=False):
        self.calls += 1
        e = self.emb[input_ids]
        steps = torch.arange(1, input_ids.shape[1] + 1).view(1, -1, 1)
        return SimpleNamespace(last_hidden_state=e.cumsum(1) / steps)


def noul(label=True):
    return {"type": "noul", "instructions": "Is it malignant?", "label": label, "src": "x"}


def choice(label="b"):
    return {"type": "choice", "instructions": "Which one?", "criteria": {"a": "first", "b": "second", "c": "third"},
            "label": label, "src": "x"}


def score(label=2):
    return {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high", "max"], "label": label,
            "src": "x"}


def test_request_question_never_carries_the_label():
    q = {**choice(), "target": {"b": 1.0}}
    assert set(ql.request_question(q)) == {"type", "instructions", "criteria"}


def test_messages_are_the_frontier_adapters():
    from serve import Frontier
    state, q = "synthetic report", choice()
    ours = ql.messages(state, "q1", q)
    theirs = Frontier.messages(SimpleNamespace(fewshot=[]), state, "q1", ql.request_question(q))
    assert ours == theirs
    assert sorted(ours[1]) == ["a", "b", "c"]


def test_letter_ids_single_tokens_and_refuses_multi_token_letters():
    assert ql.letter_ids(FakeTok(), 3) == [ord("A"), ord("B"), ord("C")]

    class TwoTok(FakeTok):
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1, 2]}
    with pytest.raises(ValueError):
        ql.letter_ids(TwoTok(), 2)


def test_encode_question_refuses_document_over_max_state():
    with pytest.raises(ContextOverflow):
        ql.encode_question(FakeTok(), "x" * 50, "q", noul(), max_state=49)
    ids, keys = ql.encode_question(FakeTok(), "x" * 49, "q", noul(), max_state=49)
    assert len(ids) > 49 and sorted(keys) == ["false", "true"]


@pytest.mark.parametrize("q", [noul(True), noul(False), choice("c"), score(3)])
def test_training_example_answer_is_the_letter_of_the_label(q):
    tok = FakeTok()
    rec = {"state": "synthetic", "questions": {"q": q}}
    [(ids, answer)] = ql.training_examples(tok, rec, max_state=100)
    _, keys = ql.messages("synthetic", "q", q)
    shown = keys.index("true" if q["label"] is True else "false" if q["label"] is False else str(q["label"]))
    assert answer == ord("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[shown])
    assert ids == tok(ql.prompt_text(tok, ql.messages("synthetic", "q", q)[0]))["input_ids"]


def test_training_examples_one_per_question_and_whole_record_dropped_when_too_long():
    rec = {"state": "short", "questions": {"a": noul(), "b": choice(), "c": score()}}
    assert len(ql.training_examples(FakeTok(), rec, max_state=100)) == 3
    assert ql.training_examples(FakeTok(), {**rec, "state": "y" * 200}, max_state=100) == []


def test_option_distribution_is_a_restricted_softmax():
    p, lp = ql.option_distribution(torch.tensor([2.0, 0.0, -1.0]))
    assert abs(sum(p) - 1) < 1e-6 and p[0] > p[1] > p[2]
    assert all(abs(torch.tensor(x).exp().item() - y) < 1e-6 for x, y in zip(lp, p))


def predictor(shots=0, max_state=100):
    bq.QwenPredictor.shots, bq.QwenPredictor.task = shots, "p2"
    return bq.QwenPredictor("base", "cpu", context={"max_state": max_state}, model=FakeCausalLM(), tok=FakeTok())


def test_predictor_returns_a_distribution_and_logits_per_question_at_temperature_one():
    rec = {"state": "synthetic", "questions": {"a": noul(), "b": choice(), "c": score()}}
    out = predictor()(rec)
    assert set(out["probabilities"]) == {"a", "b", "c"} and out["inference_temperature"] == 1.0
    assert set(out["probabilities"]["c"]) == {"0", "1", "2", "3"} and set(out["logits"]["b"]) == {"a", "b", "c"}
    for qid in out["probabilities"]:
        assert abs(sum(out["probabilities"][qid].values()) - 1) < 1e-6


def test_predictor_rejects_overlong_documents_like_kev():
    with pytest.raises(ContextOverflow):
        predictor(max_state=10)({"state": "z" * 11, "questions": {"a": noul()}})


def test_kev_benchmark_accepts_the_prediction_rows(tmp_path):
    from kev.benchmark import prediction_rows
    from kev.data import load_records
    f = tmp_path / "dev.jsonl"
    f.write_text(json.dumps({"_meta": {"id": "r1", "group": "g1"}, "state": "synthetic",
                             "questions": {"a": noul(), "b": choice(), "c": score()}}) + "\n")
    [rec] = load_records(str(f))
    rows = prediction_rows(rec, predictor()(rec))
    assert len(rows) == 3 and all("logits" in r and len(r["logits"]) == len(r["keys"]) for r in rows)
    assert all(r["variant"] == "clean" for r in rows)


def test_pop_flag_removes_our_flags_and_keeps_kev_arguments():
    argv = ["--max_state", "2048", "--task", "i1", "--run", "base", "--data", "d.jsonl"]
    assert bq.pop_flag(argv, "--max_state") == "2048" and bq.pop_flag(argv, "--shots", "0") == "0"
    assert bq.pop_flag(argv, "--task") == "i1" and argv == ["--run", "base", "--data", "d.jsonl"]


def test_epoch_order_is_a_seeded_permutation_that_changes_per_epoch():
    a, b = ts.epoch_order(100, 0, 0), ts.epoch_order(100, 0, 1)
    assert sorted(a) == list(range(100)) and a != b and a == ts.epoch_order(100, 0, 0)
    assert ts.epoch_order(100, 1, 0) != a


def test_optimizer_steps_per_epoch_round_up():
    assert ts.n_optimizer_steps(3000, 2, 8) == 750 and ts.n_optimizer_steps(41053, 2, 8) == 10264


def test_build_examples_counts_dropped_records():
    recs = [{"state": "ok", "questions": {"a": noul()}}, {"state": "w" * 500, "questions": {"a": noul()}}]
    examples, dropped = ts.build_examples(FakeTok(), recs, max_state=100)
    assert len(examples) == 1 and dropped == 1


def test_train_refuses_existing_out(tmp_path):
    (tmp_path / "run").mkdir()
    assert ts.main(["--data", "x.jsonl", "--out", str(tmp_path / "run"), "--max_state", "1024"]) == 1


def test_answer_logits_are_unchanged_by_right_padding():
    lm = FakeCausalLM()
    short, long = [65, 66, 67], [70, 71, 72, 73, 74, 75, 76]
    alone = ql.answer_logits(lm, [short], "cpu")[0]
    padded = ql.answer_logits(lm, [short, long], "cpu")[0]
    assert torch.allclose(alone, padded)
    assert torch.allclose(ql.answer_logits(lm, [long], "cpu")[0], ql.answer_logits(lm, [short, long], "cpu")[1])


def test_predictor_scores_all_questions_of_a_record_in_one_pass():
    p = predictor()
    p({"state": "synthetic", "questions": {"a": noul(), "b": choice(), "c": score()}})
    assert p.causal.calls == 1


def test_batches_cover_every_example_once():
    order = ts.epoch_order(21, 0, 0)
    bs = ts.batches(order, 8)
    assert [len(b) for b in bs] == [8, 8, 5] and sorted(sum(bs, [])) == list(range(21))


def test_train_refuses_a_subset_file_whose_hash_differs_from_its_summary(tmp_path):
    data = tmp_path / "p9-3k.jsonl"
    data.write_text('{"x": 1}\n')
    subsets = tmp_path / "subsets"; subsets.mkdir()
    (subsets / "p9-3k.json").write_text(json.dumps({"out": str(data), "out_sha256": "0" * 64}))
    assert ts.subset_mismatch(data, subsets) is not None
    (subsets / "p9-3k.json").write_text(json.dumps({"out": str(data), "out_sha256": ts._sha256(data)}))
    assert ts.subset_mismatch(data, subsets) is None
    assert ts.subset_mismatch(tmp_path / "unlisted.jsonl", subsets) is None


def test_train_main_refuses_mismatching_subset_before_loading_anything(tmp_path):
    data = tmp_path / "p9-3k.jsonl"; data.write_text('{"x": 1}\n')
    subsets = tmp_path / "subsets"; subsets.mkdir()
    (subsets / "p9-3k.json").write_text(json.dumps({"out": str(data), "out_sha256": "0" * 64}))
    assert ts.main(["--data", str(data), "--out", str(tmp_path / "run"), "--max_state", "1024",
                    "--subsets_dir", str(subsets)]) == 1
    assert not (tmp_path / "run").exists()


def test_single_sequence_batches_are_run_without_an_attention_mask():
    seen = []

    class Spy(FakeCausalLM):
        def model(self, input_ids, attention_mask, use_cache=False):
            seen.append(attention_mask)
            return super().model(input_ids, attention_mask, use_cache)
    lm = Spy()
    ql.answer_logits(lm, [[65, 66, 67]], "cpu")
    ql.answer_logits(lm, [[65, 66], [65, 66, 67]], "cpu")
    assert seen[0] is None and seen[1] is not None
