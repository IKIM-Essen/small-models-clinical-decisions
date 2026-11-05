"""Token-budget guard for every generative baseline call.

Pure functions, no network. `litellm_client.py` wires them into the request loop.
Rules implemented here:
- count state tokens before the call and fit the state with the task's declared
  strategy, so every model sees the same input and the gateway never truncates;
- max_tokens = 8 for non-thinking; a declared trace cap for thinking, doubled once
  on finish_reason == "length", then a parse failure;
- classify HTTP errors, honour retry-after on 429;
- detect silent gateway fallback via the x-litellm-model-api-base header;
- probabilities = softmax over first-token logprobs of the option letters;
- append-only JSONL ledger so a restarted run never pays twice for a call.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import deque
from pathlib import Path
from typing import Callable, Iterable

CHARS_PER_TOKEN = 3  # conservative: typical BPE is ~4 chars/token, so we over-count
NON_THINKING_MAX_TOKENS = 8
BACKOFF_CAP_S = 60.0
TERMINAL_STATUSES = {"ok", "parse_failure"}
MAX_FALLBACK_RERUNS = 3  # after this many discarded fallback answers the call is terminal
_SPECIAL = re.compile(r"<\|[^|]*\|>")


class ContextBudgetError(ValueError):
    """Prompt plus output budget does not fit the model's context window."""


# --- tokens and context strategy --------------------------------------------

def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def fit_state(text: str, budget_tokens: int, strategy: str) -> str:
    """Fit `text` into `budget_tokens` with the declared strategy.

    `truncate` keeps the head, `head_tail` keeps equal parts of both ends.
    Section-aware truncation and chunk-and-aggregate are task-level and live in the
    task builder; they hand this function an already ordered text.
    """
    if strategy not in {"truncate", "head_tail"}:
        raise ValueError(f"unknown context strategy: {strategy}")
    if estimate_tokens(text) <= budget_tokens:
        return text
    max_chars = budget_tokens * CHARS_PER_TOKEN
    if strategy == "truncate":
        return text[:max_chars]
    sep = " [...] "
    half = (max_chars - len(sep)) // 2
    return text[:half] + sep + text[-half:]


def max_tokens_for(thinking: bool, trace_cap: int = 4096, attempt: int = 0) -> int:
    if not thinking:
        return NON_THINKING_MAX_TOKENS
    if attempt not in (0, 1):
        raise ValueError("thinking calls get one retry with a doubled cap, no more")
    return trace_cap * (2 ** attempt)


def check_budget(prompt_tokens: int, max_tokens: int, context_limit: int) -> None:
    if prompt_tokens + max_tokens > context_limit:
        raise ContextBudgetError(
            f"prompt {prompt_tokens} + max_tokens {max_tokens} > context {context_limit}"
        )


def model_limits(model_info: dict) -> dict:
    """Reduce a LiteLLM /model/info response to {model: {context, max_output, api_base}}."""
    out = {}
    for entry in model_info.get("data", []):
        info = entry.get("model_info") or {}
        params = entry.get("litellm_params") or {}
        out[entry["model_name"]] = {
            "context": info.get("max_input_tokens") or info.get("max_tokens"),
            "max_output": info.get("max_output_tokens"),
            "api_base": params.get("api_base"),
        }
    return out


# --- errors and retry --------------------------------------------------------

def classify_http(status: int, body: str) -> str:
    text = (body or "").lower()
    if status == 200:
        return "ok"
    if status == 429:
        return "rate_limit"
    if status in (401, 403):
        return "auth"
    if status == 400 and ("context length" in text or "contextwindowexceeded" in text
                          or "context_length" in text or "too many tokens" in text):
        return "context_length"
    if status >= 500:
        return "server"
    return "bad_request"


def retry_delay(headers: dict, attempt: int, base: float = 2.0, jitter: float = 1.0,
                rng: random.Random | None = None) -> float:
    for k, v in headers.items():
        if k.lower() == "retry-after":
            try:
                return float(v)
            except ValueError:
                break
    delay = min(BACKOFF_CAP_S, base * (2 ** attempt))
    if jitter:
        delay = min(BACKOFF_CAP_S, delay + (rng or random).uniform(0, jitter))
    return delay


def hit_length_cap(response: dict) -> bool:
    choices = response.get("choices") or [{}]
    return choices[0].get("finish_reason") == "length"


def served_by_fallback(headers: dict, expected_api_base) -> bool:
    """True if the gateway answered from another backend (or we cannot tell).

    `expected_api_base` is one URL or a set of URLs (a load-balanced pool; take all
    `api_base` values of the model from /model/info).
    """
    served = next((v for k, v in headers.items() if k.lower() == "x-litellm-model-api-base"), None)
    if not served:
        return True
    norm = lambda u: u.rstrip("/").removesuffix("/v1")
    expected = {expected_api_base} if isinstance(expected_api_base, str) else set(expected_api_base)
    return norm(served) not in {norm(e) for e in expected}


# --- probabilities -----------------------------------------------------------

def _clean(token: str) -> str:
    return _SPECIAL.sub("", token).strip()


def letter_probs(top_logprobs: dict, letters: Iterable[str], first_token: str) -> dict | None:
    """Softmax over option-letter logprobs of the first generated token.

    Returns None (parse failure) when the first token is not an option letter.
    """
    letters = list(letters)
    if _clean(first_token) not in letters:
        return None
    if not any(_clean(t) in letters for t in top_logprobs):
        return None
    best = {}
    for tok, lp in top_logprobs.items():
        c = _clean(tok)
        if c in letters:
            best[c] = max(best.get(c, -math.inf), lp)
    m = max(best.values())
    exp = {k: math.exp(v - m) for k, v in best.items()}
    z = sum(exp.values())
    return {k: exp.get(k, 0.0) / z for k in letters}


def extract_letter(content: str, letters: Iterable[str]) -> str | None:
    """Final option letter from a thinking-mode answer (label only, no probability)."""
    letters = list(letters)
    alt = "|".join(re.escape(x) for x in letters)
    found = re.findall(rf"(?<![A-Za-z])({alt})(?![A-Za-z])", _clean(content))
    return found[-1] if found else None


# --- ledger ------------------------------------------------------------------

def request_key(payload: dict) -> str:
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode()).hexdigest()


def ledger_append(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def ledger_done(path: Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    done, fallbacks = set(), {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # truncated last line after a crash
        if rec.get("status") in TERMINAL_STATUSES:
            done.add(rec["key"])
        elif rec.get("status") == "fallback_discarded":
            fallbacks[rec["key"]] = fallbacks.get(rec["key"], 0) + 1
            if fallbacks[rec["key"]] >= MAX_FALLBACK_RERUNS:
                done.add(rec["key"])  # reported as unresolved fallback, never paid again
    return done


# --- client-side rate limiter ------------------------------------------------

class RateLimiter:
    """Sliding 60 s window over requests and tokens, kept below the key's limits."""

    def __init__(self, tpm: int, rpm: int, clock: Callable[[], float]):
        self.tpm, self.rpm, self.clock = tpm, rpm, clock
        self.events: deque[tuple[float, int]] = deque()

    def _prune(self, now: float) -> None:
        while self.events and now - self.events[0][0] >= 60.0:
            self.events.popleft()

    def wait_for(self, tokens: int) -> float:
        now = self.clock()
        self._prune(now)
        used = sum(t for _, t in self.events)
        if len(self.events) < self.rpm and used + tokens <= self.tpm:
            return 0.0
        return max(0.0, 60.0 - (now - self.events[0][0])) if self.events else 0.0

    def record(self, tokens: int) -> None:
        self.events.append((self.clock(), tokens))
