"""The client is tested against a fake transport: no network, no secrets."""
import pytest

from conftest import load_script

load_script("scripts/llm/token_guard.py")
lc = load_script("scripts/llm/litellm_client.py")

DS_BASE = "http://ds.svc/v1"


class FakeTransport:
    """Scripted responses; records every request (headers included) for assertions."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def __call__(self, method, url, headers, body):
        self.calls.append({"method": method, "url": url, "headers": dict(headers), "body": body})
        status, resp_headers, resp_body = self.script.pop(0)
        return status, resp_headers, resp_body


def ok(content="A", api_base=DS_BASE, finish="stop"):
    return (
        200,
        {"x-litellm-model-api-base": api_base},
        {"choices": [{"message": {"content": content}, "finish_reason": finish,
                      "logprobs": {"content": [{"token": content, "top_logprobs": [
                          {"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -2.5}]}]}}],
         "usage": {"prompt_tokens": 50, "completion_tokens": 1}},
    )


def token_resp():
    return (200, {}, {"access_token": "sso-token-xyz", "expires_in": 86400})


def make_client(transport, env=None, sleeps=None):
    env = env if env is not None else {"KEV_LITELLM_KEY": "sk-virtual", "SSO_USERNAME": "u", "SSO_PASSWORD": "p"}
    sleeps = sleeps if sleeps is not None else []
    return lc.LiteLLMClient(
        base_url="https://gw/v1",
        env=env,
        transport=transport,
        expected_api_base={"deepseek-v4-flash-0731": DS_BASE},
        context_limit={"deepseek-v4-flash-0731": 384000},
        sleep=sleeps.append,
        clock=lambda: 0.0,
    )


def payload():
    return {"model": "deepseek-v4-flash-0731", "messages": [{"role": "user", "content": "Q? A or B"}],
            "max_tokens": 8, "logprobs": True, "top_logprobs": 5, "temperature": 0}


def test_uses_virtual_key_first():
    t = FakeTransport([ok()])
    r = make_client(t).chat(payload())
    assert r["status"] == "ok"
    assert t.calls[0]["headers"]["Authorization"] == "Bearer sk-virtual"
    assert r["credential"] == "virtual_key"


def test_falls_back_to_sso_on_auth_error(monkeypatch):
    monkeypatch.setattr(lc, "SSO_TOKEN_URL", "https://sso.example.org/realms/r/protocol/openid-connect/token")
    t = FakeTransport([(401, {}, {"error": "bad key"}), token_resp(), ok()])
    r = make_client(t).chat(payload())
    assert r["status"] == "ok" and r["credential"] == "sso"
    assert "openid-connect/token" in t.calls[1]["url"]
    assert t.calls[2]["headers"]["Authorization"] == "Bearer sso-token-xyz"


def test_no_credentials_raises():
    with pytest.raises(lc.CredentialError):
        make_client(FakeTransport([]), env={})


def test_rate_limit_honours_retry_after_then_succeeds():
    sleeps = []
    t = FakeTransport([(429, {"retry-after": "5"}, {"error": "rl"}), ok()])
    r = make_client(t, sleeps=sleeps).chat(payload())
    assert r["status"] == "ok"
    assert 5.0 in sleeps


def test_silent_fallback_is_discarded_not_returned_as_ok():
    t = FakeTransport([ok(api_base="http://qwen36.svc/v1")])
    r = make_client(t).chat(payload())
    assert r["status"] == "fallback_discarded"
    assert r["served_by"] == "http://qwen36.svc/v1"


def test_context_length_error_is_reported_not_retried():
    t = FakeTransport([(400, {}, {"error": {"message": "maximum context length exceeded"}})])
    r = make_client(t).chat(payload())
    assert r["status"] == "context_length"
    assert len(t.calls) == 1


def test_prompt_over_budget_is_rejected_before_any_call():
    t = FakeTransport([])
    c = make_client(t)
    c.context_limit["deepseek-v4-flash-0731"] = 10
    r = c.chat(payload())
    assert r["status"] == "context_length" and t.calls == []


def test_thinking_length_cap_retries_once_with_double_cap():
    p = payload()
    p["max_tokens"] = 100
    t = FakeTransport([ok(finish="length"), ok(finish="length")])
    r = make_client(t).chat(p, thinking=True, trace_cap=100)
    assert [c["body"]["max_tokens"] for c in t.calls] == [100, 200]
    assert r["status"] == "parse_failure"


def test_probabilities_come_from_first_token_logprobs():
    t = FakeTransport([ok()])
    r = make_client(t).chat(payload(), letters=["A", "B"])
    assert r["answer"] == "A"
    assert r["probs"]["A"] > r["probs"]["B"]


def test_secrets_never_in_result():
    t = FakeTransport([(401, {}, {}), token_resp(), ok()])
    r = make_client(t).chat(payload())
    flat = repr(r)
    assert "sk-virtual" not in flat and "sso-token-xyz" not in flat and "'p'" not in flat


def test_network_error_is_retried_as_server_error():
    import urllib.error
    calls = []

    def flaky(method, url, headers, body):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.URLError("connection reset")
        return ok()

    r = make_client(flaky).chat(payload())
    assert r["status"] == "ok" and len(calls) == 2


def test_missing_logprobs_is_parse_failure_not_crash():
    resp = (200, {"x-litellm-model-api-base": DS_BASE},
            {"choices": [{"message": {"content": "A"}, "finish_reason": "stop", "logprobs": {"content": [
                {"token": "A", "top_logprobs": []}]}}]})
    r = make_client(FakeTransport([resp])).chat(payload(), letters=["A", "B"])
    assert r["status"] == "parse_failure"


def test_thinking_answer_letter_is_extracted():
    resp = ok(content="After reasoning, the answer is B.")
    r = make_client(FakeTransport([resp])).chat(payload(), thinking=True, trace_cap=100,
                                                 letters_answer_only=["A", "B"])
    assert r["answer"] == "B"
