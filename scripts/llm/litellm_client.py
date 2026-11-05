"""Budget-safe client for the internal LiteLLM gateway.

Skeleton: the request loop, auth fallback and guard rules are complete and tested
against a fake transport; the task runner that builds prompts and writes the ledger
is scripts/frontier/serve.py. Secrets are read from env and never returned, logged or printed.

Auth: project virtual key KEV_LITELLM_KEY first; on 401/403 fall back to a personal
SSO token (Keycloak password grant, client litellm-proxy, valid 24 h, re-minted
1 h before expiry) from SSO_USERNAME / SSO_PASSWORD.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import token_guard as tg  # noqa: E402

SSO_TOKEN_URL = os.environ.get("SSO_TOKEN_URL", "")   # Keycloak token endpoint, if the gateway uses SSO
SSO_REFRESH_MARGIN_S = 3600
MAX_RATE_RETRIES = 6
MAX_SERVER_RETRIES = 3

Transport = Callable[[str, str, dict, object], tuple]


class CredentialError(RuntimeError):
    pass


def urllib_transport(method: str, url: str, headers: dict, body) -> tuple:
    """Default transport. JSON bodies for the API, form bodies for the SSO endpoint."""
    if isinstance(body, dict) and headers.get("Content-Type") == "application/x-www-form-urlencoded":
        data = urllib.parse.urlencode(body).encode()
    else:
        data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, dict(r.headers), json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raw = e.read() or b"{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"error": raw.decode(errors="replace")[:500]}
        return e.code, dict(e.headers or {}), payload


class LiteLLMClient:
    def __init__(self, base_url: str, env: dict, transport: Transport = urllib_transport,
                 expected_api_base: dict | None = None, context_limit: dict | None = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.time,
                 limiter: tg.RateLimiter | None = None):
        self.base_url = base_url.rstrip("/")
        self._vkey = env.get("KEV_LITELLM_KEY")
        self._sso_user = env.get("SSO_USERNAME")
        self._sso_pass = env.get("SSO_PASSWORD")
        if not self._vkey and not (self._sso_user and self._sso_pass):
            raise CredentialError("set KEV_LITELLM_KEY or SSO_USERNAME/SSO_PASSWORD in .env")
        self.transport = transport
        self.expected_api_base = expected_api_base or {}
        self.context_limit = context_limit or {}
        self.sleep, self.clock = sleep, clock
        self.limiter = limiter
        self._sso_token: str | None = None
        self._sso_expiry = 0.0
        self.credential = "virtual_key" if self._vkey else "sso"

    # -- auth ------------------------------------------------------------------
    def _mint_sso(self) -> None:
        if not (self._sso_user and self._sso_pass):
            raise CredentialError("virtual key rejected and no SSO credentials available")
        status, _, body = self.transport(
            "POST", SSO_TOKEN_URL, {"Content-Type": "application/x-www-form-urlencoded"},
            {"client_id": "litellm-proxy", "grant_type": "password",
             "username": self._sso_user, "password": self._sso_pass},
        )
        if status != 200 or "access_token" not in body:
            raise CredentialError(f"SSO token request failed with HTTP {status}")
        self._sso_token = body["access_token"]
        self._sso_expiry = self.clock() + float(body.get("expires_in", 86400))

    def _auth_header(self) -> dict:
        if self.credential == "virtual_key":
            return {"Authorization": f"Bearer {self._vkey}"}
        if self._sso_token is None or self.clock() > self._sso_expiry - SSO_REFRESH_MARGIN_S:
            self._mint_sso()
        return {"Authorization": f"Bearer {self._sso_token}"}

    # -- one chat call under the guard ------------------------------------------
    def chat(self, payload: dict, thinking: bool = False, trace_cap: int = 4096,
             letters: list[str] | None = None, letters_answer_only: list[str] | None = None) -> dict:
        """Returns a ledger-ready record (no secrets, no prompt text).

        `letters`: probability run, softmax over first-token logprobs (non-thinking).
        `letters_answer_only`: label only, last option letter in the answer (thinking rows).
        """
        model = payload["model"]
        key = tg.request_key(payload)
        record = {"key": key, "model": model, "credential": None, "served_by": None}
        prompt_tokens = tg.estimate_tokens(json.dumps(payload.get("messages", []), ensure_ascii=False))
        limit = self.context_limit.get(model)
        attempt_cap = 0
        body = dict(payload)
        body["max_tokens"] = tg.max_tokens_for(thinking, trace_cap, 0) if thinking else payload.get(
            "max_tokens", tg.NON_THINKING_MAX_TOKENS)
        rate_tries = server_tries = 0
        auth_switched = False
        while True:
            if limit is not None:
                try:
                    tg.check_budget(prompt_tokens, body["max_tokens"], limit)
                except tg.ContextBudgetError as e:
                    return {**record, "status": "context_length", "detail": str(e)}
            if self.limiter is not None:
                wait = self.limiter.wait_for(prompt_tokens + body["max_tokens"])
                if wait:
                    self.sleep(wait)
            t0 = self.clock()
            try:
                status, headers, resp = self.transport(
                    "POST", f"{self.base_url}/chat/completions",
                    {"Content-Type": "application/json", **self._auth_header()}, body)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                status, headers, resp = 599, {}, {"error": type(e).__name__}
            if self.limiter is not None:
                self.limiter.record(prompt_tokens + body["max_tokens"])
            record["credential"] = self.credential
            cls = tg.classify_http(status, json.dumps(resp)[:2000])
            if cls == "auth" and self.credential == "virtual_key" and not auth_switched:
                self.credential, auth_switched = "sso", True
                continue
            if cls == "rate_limit" and rate_tries < MAX_RATE_RETRIES:
                self.sleep(tg.retry_delay(headers, rate_tries))
                rate_tries += 1
                continue
            if cls == "server" and server_tries < MAX_SERVER_RETRIES:
                self.sleep(tg.retry_delay(headers, server_tries))
                server_tries += 1
                continue
            if cls != "ok":
                return {**record, "status": cls, "http": status}

            record["served_by"] = next(
                (v for k, v in headers.items() if k.lower() == "x-litellm-model-api-base"), None)
            record["latency_s"] = self.clock() - t0
            record["usage"] = resp.get("usage")
            expected = self.expected_api_base.get(model)
            if expected is not None and tg.served_by_fallback(headers, expected):
                return {**record, "status": "fallback_discarded"}
            if thinking and tg.hit_length_cap(resp):
                if attempt_cap == 0:
                    attempt_cap = 1
                    body = {**body, "max_tokens": tg.max_tokens_for(True, trace_cap, 1)}
                    continue
                return {**record, "status": "parse_failure", "detail": "length cap after retry"}
            return {**record, **self._parse(resp, letters, letters_answer_only)}

    @staticmethod
    def _parse(resp: dict, letters: list[str] | None, letters_answer_only: list[str] | None = None) -> dict:
        choice = (resp.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        out = {"status": "ok", "answer": tg._clean(content)[:16]}
        if letters_answer_only:
            letter = tg.extract_letter(content, letters_answer_only)
            if letter is None:
                return {**out, "status": "parse_failure", "detail": "no option letter in answer"}
            return {**out, "answer": letter}
        if letters:
            lp = ((choice.get("logprobs") or {}).get("content") or [])
            if not lp:
                return {**out, "status": "parse_failure", "detail": "no logprobs"}
            first = lp[0]
            top = {t["token"]: t["logprob"] for t in first.get("top_logprobs", [])}
            probs = tg.letter_probs(top, letters, first.get("token", ""))
            if probs is None:
                return {**out, "status": "parse_failure", "detail": "first token not an option letter"}
            out["probs"] = probs
            out["answer"] = max(probs, key=probs.get)
        return out
