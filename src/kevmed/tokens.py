"""Token-length statistics per split: aggregate histograms only."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from kev.api import render
from kev.model import load_tokenizer as _kev_tok

BINS = [0, 128, 256, 384, 512, 768, 1024, 2048, 4096, 7552]
DEFAULT_BASE = "Qwen/Qwen3.5-4B-Base"


def load_tokenizer(base: str = DEFAULT_BASE):
    return _kev_tok(base)


def state_text(state) -> str:
    return render(state)


def n_tokens(text: str, tok) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def _pct(xs: list[int], p: float) -> int:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def histogram(lengths: list[int], bins: list[int] = BINS) -> dict:
    if not lengths:
        return {"n": 0, "min": None, "p50": None, "p90": None, "p95": None, "max": None, "mean": None,
                "bins": {}, "over_384": 0, "over_1024": 0, "over_7552": 0}
    counts = {}
    for lo, hi in zip(bins, bins[1:] + [None]):
        label = f"{lo}-{hi}" if hi else f"{lo}+"
        counts[label] = sum(1 for x in lengths if x >= lo and (hi is None or x < hi))
    return {"n": len(lengths), "min": min(lengths), "p50": _pct(lengths, 0.5), "p90": _pct(lengths, 0.9),
            "p95": _pct(lengths, 0.95), "max": max(lengths), "mean": round(statistics.fmean(lengths), 1),
            "bins": counts, "over_384": sum(x > 384 for x in lengths), "over_1024": sum(x > 1024 for x in lengths),
            "over_7552": sum(x > 7552 for x in lengths)}


def split_stats(path: Path, tok) -> dict:
    states, qmax = [], 0
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            states.append(n_tokens(state_text(r["state"]), tok))
            for q in r["questions"].values():
                qtext = render(q.get("instructions")) + "\n" + render(q.get("criteria"))
                qmax = max(qmax, n_tokens(qtext, tok))
    return {"file": Path(path).name, "state": histogram(states), "question_max": qmax}
