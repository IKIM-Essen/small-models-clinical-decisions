# /// script
# requires-python = ">=3.11"
# ///
"""Qwen3.5-4B Instruct as a System One predictor and as the LoRA-SFT baseline.

Prompts are the frontier adapter's own (scripts/frontier/serve.py, `Frontier.messages`): the same system prompt, the
same lettered options shuffled per example with the same seed, the same letter-only instruction, the same few-shot
turns. So the size-matched tier sees exactly what the frontier tier sees. The probability of an option is the softmax
over the option letters' next-token logits at the answer position (all options: the full vocabulary logits are local,
so I2's 22 chapters need no top-k workaround). A document longer than `max_state` tokens is refused with kev's
ContextOverflow, so kev.benchmark counts it as rejected exactly as it does for Kev.
Never prints states or prompts."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "frontier"))
import serve as frontier  # noqa: E402  (imports only; no network on import)

from kev.model import ContextOverflow  # noqa: E402

BASE = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
REQUEST_KEYS = ("type", "instructions", "criteria")   # what a /v1/systemone request carries: never labels


def request_question(q: dict) -> dict:
    return {k: v for k, v in q.items() if k in REQUEST_KEYS}


def messages(state, qid: str, q: dict, fewshot=()) -> tuple[list[dict], list[str]]:
    """Chat messages and the option keys in the order shown (A, B, ...), via the frontier adapter's own code."""
    return frontier.Frontier.messages(SimpleNamespace(fewshot=list(fewshot)), state, qid, request_question(q))


def prompt_text(tok, msgs: list[dict]) -> str:
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def ids_of(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def letter_ids(tok, n: int) -> list[int]:
    out = []
    for letter in frontier.LETTERS[:n]:
        ids = ids_of(tok, letter)
        if len(ids) != 1:
            raise ValueError(f"option letter {letter!r} is not a single token")
        out.append(ids[0])
    return out


def document_tokens(tok, state) -> int:
    """Tokens of the document as the prompt shows it (`Document:\\n{state}`)."""
    return len(ids_of(tok, f"{state}"))


def encode_question(tok, state, qid: str, q: dict, max_state: int, fewshot=()) -> tuple[list[int], list[str]]:
    n = document_tokens(tok, state)
    if n > max_state:
        raise ContextOverflow(f"state exceeds {max_state} tokens: {n}")
    msgs, keys = messages(state, qid, q, fewshot)
    return ids_of(tok, prompt_text(tok, msgs)), keys


def answer_letter_id(tok, keys: list[str], q: dict) -> int:
    """Token id of the letter under which the labelled answer is shown."""
    return letter_ids(tok, len(keys))[keys.index(frontier.label_key(q))]


def training_examples(tok, record: dict, max_state: int) -> list[tuple[list[int], int]]:
    """One (prompt ids, answer letter id) per question of a labelled record; [] if the document is over max_state."""
    out = []
    for qid, q in record["questions"].items():
        try:
            ids, keys = encode_question(tok, record["state"], qid, q, max_state)
        except ContextOverflow:
            return []
        out.append((ids, answer_letter_id(tok, keys, q)))
    return out


def answer_logits(causal_lm, batch: list[list[int]], device, pad_id: int = 0):
    """Next-token logits at each sequence's own last position, for a batch of prompts of different lengths.

    Right padding only: the models are causal (attention and the recurrent Gated DeltaNet layers alike), so pads after a
    sequence's last real token cannot change any of its positions; left padding would. Only the hidden state at the
    answer position goes through the LM head, so long letters never materialise full-vocabulary logits per token.
    `causal_lm` is a *ForCausalLM (or the PEFT-wrapped one's base model) with `.model` and `.lm_head`."""
    import torch
    lengths = torch.tensor([len(x) for x in batch], device=device)
    ids = torch.full((len(batch), int(lengths.max())), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, x in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x, device=device)
        mask[i, :len(x)] = 1
    # a single sequence has no padding: no mask, so SDPA can use its memory-efficient kernel instead of the math
    # kernel's full L x L matrix (I2 with three few-shot letters is about 30k tokens; OOM in fp32 with a mask)
    hidden = causal_lm.model(input_ids=ids, attention_mask=None if len(batch) == 1 else mask,
                             use_cache=False).last_hidden_state
    last = hidden[torch.arange(len(batch), device=device), lengths - 1]
    return causal_lm.lm_head(last)


def option_distribution(letter_logits) -> tuple[list[float], list[float]]:
    """Restricted softmax over the option letters: (probabilities, log-probabilities as logits)."""
    import torch
    logp = torch.log_softmax(letter_logits.float(), dim=-1)
    return logp.exp().tolist(), logp.tolist()
