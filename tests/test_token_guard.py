import json
import math

import pytest

from conftest import load_script

tg = load_script("scripts/llm/token_guard.py")


# --- token estimate and context strategy ---------------------------------

def test_estimate_tokens_is_conservative():
    # 3 chars per token upper bound: never under-counts relative to typical BPE (~4 chars/token)
    assert tg.estimate_tokens("a" * 300) == 100
    assert tg.estimate_tokens("") == 0
    assert tg.estimate_tokens("abcd") == 2  # ceil(4/3)


def test_fit_state_leaves_short_text_untouched():
    text = "short state"
    assert tg.fit_state(text, budget_tokens=100, strategy="truncate") == text


def test_fit_state_truncate_keeps_head_within_budget():
    text = "x" * 3000
    out = tg.fit_state(text, budget_tokens=100, strategy="truncate")
    assert text.startswith(out)
    assert tg.estimate_tokens(out) <= 100


def test_fit_state_head_tail_keeps_both_ends():
    text = "H" * 1500 + "M" * 3000 + "T" * 1500
    out = tg.fit_state(text, budget_tokens=200, strategy="head_tail")
    assert out.startswith("H") and out.endswith("T")
    assert tg.estimate_tokens(out) <= 200


def test_fit_state_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        tg.fit_state("x" * 3000, budget_tokens=10, strategy="magic")


# --- max_tokens and context budget ---------------------------------------

def test_max_tokens_non_thinking_is_tiny():
    assert tg.max_tokens_for(thinking=False) == 8


def test_max_tokens_thinking_doubles_once_on_retry():
    assert tg.max_tokens_for(thinking=True, trace_cap=4096, attempt=0) == 4096
    assert tg.max_tokens_for(thinking=True, trace_cap=4096, attempt=1) == 8192
    with pytest.raises(ValueError):
        tg.max_tokens_for(thinking=True, trace_cap=4096, attempt=2)


def test_check_budget_raises_before_the_call():
    tg.check_budget(prompt_tokens=1000, max_tokens=8, context_limit=2000)
    with pytest.raises(tg.ContextBudgetError):
        tg.check_budget(prompt_tokens=1995, max_tokens=8, context_limit=2000)


# --- error classes and retry --------------------------------------------

@pytest.mark.parametrize(
    "status,body,expected",
    [
        (200, "", "ok"),
        (429, "rate limit", "rate_limit"),
        (400, "This model's maximum context length is 262144 tokens", "context_length"),
        (400, "ContextWindowExceededError", "context_length"),
        (400, "bad param", "bad_request"),
        (401, "", "auth"),
        (403, "", "auth"),
        (502, "", "server"),
        (503, "overloaded", "server"),
    ],
)
def test_classify_http(status, body, expected):
    assert tg.classify_http(status, body) == expected


def test_retry_delay_honours_retry_after():
    assert tg.retry_delay({"retry-after": "7"}, attempt=0) == 7.0
    assert tg.retry_delay({"Retry-After": "3"}, attempt=5) == 3.0


def test_retry_delay_backoff_is_capped_and_grows():
    d0 = tg.retry_delay({}, attempt=0, jitter=0.0)
    d3 = tg.retry_delay({}, attempt=3, jitter=0.0)
    d20 = tg.retry_delay({}, attempt=20, jitter=0.0)
    assert d0 < d3 <= 60.0 and d20 == 60.0


def test_finish_reason_length_is_detected():
    assert tg.hit_length_cap({"choices": [{"finish_reason": "length"}]})
    assert not tg.hit_length_cap({"choices": [{"finish_reason": "stop"}]})


# --- silent fallback detection -------------------------------------------

def test_fallback_detected_from_api_base_header():
    expected = "http://deepseek-v4-flash-0731.svc:8000"
    assert not tg.served_by_fallback({"x-litellm-model-api-base": expected + "/v1"}, expected)
    assert tg.served_by_fallback({"x-litellm-model-api-base": "http://qwen36.svc:8000/v1"}, expected)
    # missing header: cannot prove who answered -> treat as fallback (discard, rerun)
    assert tg.served_by_fallback({}, expected)


# --- option-letter probabilities ----------------------------------------

def test_letter_probs_softmax_over_option_letters():
    top = {"A": math.log(0.6), "B": math.log(0.2), "C": math.log(0.1), "The": math.log(0.1)}
    p = tg.letter_probs(top, letters=["A", "B", "C"], first_token="A")
    assert p is not None
    assert abs(sum(p.values()) - 1.0) < 1e-9
    assert abs(p["A"] - 0.6 / 0.9) < 1e-9


def test_letter_probs_parse_failure_when_first_token_not_a_letter():
    top = {"The": -0.1, "A": -3.0}
    assert tg.letter_probs(top, letters=["A", "B"], first_token="The") is None


def test_letter_probs_strips_whitespace_and_special_tokens():
    top = {" A": math.log(0.5), "B<|close|>": math.log(0.5)}
    p = tg.letter_probs(top, letters=["A", "B"], first_token=" A")
    assert p == pytest.approx({"A": 0.5, "B": 0.5})


# --- ledger resume -------------------------------------------------------

def test_request_key_is_order_independent():
    a = {"model": "m", "messages": [{"role": "user", "content": "x"}], "max_tokens": 8}
    b = {"max_tokens": 8, "messages": [{"content": "x", "role": "user"}], "model": "m"}
    assert tg.request_key(a) == tg.request_key(b)


def test_ledger_resume_skips_terminal_records_only(tmp_path):
    ledger = tmp_path / "calls.jsonl"
    tg.ledger_append(ledger, {"key": "k1", "status": "ok"})
    tg.ledger_append(ledger, {"key": "k2", "status": "parse_failure"})
    tg.ledger_append(ledger, {"key": "k3", "status": "rate_limit"})  # retryable, not done
    tg.ledger_append(ledger, {"key": "k4", "status": "fallback_discarded"})  # must rerun
    assert tg.ledger_done(ledger) == {"k1", "k2"}


def test_ledger_tolerates_truncated_last_line(tmp_path):
    ledger = tmp_path / "calls.jsonl"
    tg.ledger_append(ledger, {"key": "k1", "status": "ok"})
    with ledger.open("a") as f:
        f.write('{"key": "k2", "sta')  # crash mid-write
    assert tg.ledger_done(ledger) == {"k1"}


def test_ledger_missing_file_is_empty(tmp_path):
    assert tg.ledger_done(tmp_path / "nope.jsonl") == set()


# --- rate limiter --------------------------------------------------------

def test_rate_limiter_waits_when_tpm_window_full():
    now = [0.0]
    lim = tg.RateLimiter(tpm=1000, rpm=100, clock=lambda: now[0])
    assert lim.wait_for(600) == 0.0
    lim.record(600)
    w = lim.wait_for(600)
    assert 0.0 < w <= 60.0
    now[0] = 61.0
    assert lim.wait_for(600) == 0.0


def test_rate_limiter_waits_when_rpm_full():
    now = [0.0]
    lim = tg.RateLimiter(tpm=10**9, rpm=2, clock=lambda: now[0])
    lim.record(1)
    lim.record(1)
    assert lim.wait_for(1) > 0.0


def test_model_limits_from_model_info():
    info = {
        "data": [
            {"model_name": "kimi-k3", "model_info": {"max_input_tokens": 512000, "max_output_tokens": 32768},
             "litellm_params": {"api_base": "http://kimi.svc/v1"}},
            {"model_name": "deepseek-v4-flash-0731", "model_info": {"max_tokens": 384000},
             "litellm_params": {"api_base": "http://ds.svc/v1"}},
        ]
    }
    lim = tg.model_limits(info)
    assert lim["kimi-k3"]["context"] == 512000
    assert lim["deepseek-v4-flash-0731"]["context"] == 384000
    assert lim["kimi-k3"]["api_base"] == "http://kimi.svc/v1"


def test_letter_probs_empty_top_logprobs_is_parse_failure():
    assert tg.letter_probs({}, letters=["A", "B"], first_token="A") is None


def test_fallback_accepts_any_backend_of_a_pool():
    pool = {"http://q1.svc:8000", "http://q2.svc:8000"}
    assert not tg.served_by_fallback({"x-litellm-model-api-base": "http://q2.svc:8000/v1"}, pool)
    assert tg.served_by_fallback({"x-litellm-model-api-base": "http://ds.svc/v1"}, pool)


def test_repeated_fallback_becomes_terminal(tmp_path):
    ledger = tmp_path / "calls.jsonl"
    for _ in range(tg.MAX_FALLBACK_RERUNS):
        tg.ledger_append(ledger, {"key": "k", "status": "fallback_discarded"})
    assert tg.ledger_done(ledger) == {"k"}


def test_extract_letter_from_thinking_answer():
    assert tg.extract_letter("The answer is B.", ["A", "B", "C"]) == "B"
    assert tg.extract_letter("B", ["A", "B"]) == "B"
    assert tg.extract_letter("**Answer: C**", ["A", "B", "C"]) == "C"
    assert tg.extract_letter("I cannot decide", ["A", "B"]) is None
