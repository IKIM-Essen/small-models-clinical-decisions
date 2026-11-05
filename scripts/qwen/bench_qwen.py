# /// script
# requires-python = ">=3.11"
# ///
"""Kev.benchmark --data with Qwen3.5-4B Instruct as the predictor: zero/few shot (`--run base`) or with a LoRA-SFT
adapter (`--run <adapter dir>`). kev.benchmark's own report, rows.json, metrics and coverage accounting are used
unchanged; only its LocalPredictor is swapped and its context set to the training context, as scripts/analysis/bench_ctx.py
does for Kev. Scored in fp32 with TF32 off, as kev scores Kev. Rows carry logits (log-probabilities over the options) and inference_temperature 1.0, so kev.calibrate
fits a temperature for SFT exactly as for Kev.

Usage: uv run --no-sync python scripts/qwen/bench_qwen.py --max_state N --task <t> [--shots k] --run <base|dir>
       <kev.benchmark arguments>"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_lib as ql  # noqa: E402


def pop_flag(argv: list[str], name: str, default=None):
    if name not in argv:
        return default
    i = argv.index(name)
    value = argv[i + 1]
    del argv[i:i + 2]
    return value


class QwenPredictor:
    """Drop-in for kev.predictors.LocalPredictor: __call__(record) -> probabilities and logits per question."""
    shots = 0
    task = None

    def __init__(self, run, device, opts=None, context=None, model=None, tok=None):
        import torch
        self.device = device
        self.max_state = context["max_state"]
        self.temperature = 1.0
        if model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(ql.BASE, revision=ql.BASE_REVISION)
            if device == "cuda":   # scored like kev's LocalPredictor: fp32 weights, TF32 off (Kev and SFT on equal terms)
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            model = AutoModelForCausalLM.from_pretrained(ql.BASE, revision=ql.BASE_REVISION, dtype=torch.float32)
            if run != "base":
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, run)
            model = model.to(device).eval()
        self.model, self.tok = model, tok
        self.causal = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.pad_id = getattr(tok, "pad_token_id", None) or 0
        self.fewshot = ql.frontier.load_fewshot(self.task, self.shots) if self.shots else []
        self._letters = {}

    def letters(self, n):
        if n not in self._letters:
            self._letters[n] = ql.letter_ids(self.tok, n)
        return self._letters[n]

    def __call__(self, record):
        import torch
        probs, logits = {}, {}
        encoded = {qid: ql.encode_question(self.tok, record["state"], qid, q, self.max_state, self.fewshot)
                   for qid, q in record["questions"].items()}
        start = time.perf_counter()
        with torch.no_grad():   # all questions of a record in one right-padded forward pass
            z = ql.answer_logits(self.causal, [ids for ids, _ in encoded.values()], self.device, self.pad_id)
        for row, (qid, (ids, keys)) in zip(z, encoded.items()):
            p, lp = ql.option_distribution(row[self.letters(len(keys))])
            probs[qid], logits[qid] = dict(zip(keys, p)), dict(zip(keys, lp))
        n_tokens = sum(len(ids) for ids, _ in encoded.values())
        return {"probabilities": probs, "logits": logits, "inference_temperature": 1.0,
                "latency_ms": 1000 * (time.perf_counter() - start), "input_tokens": n_tokens}


def main(argv: list[str]) -> None:
    import kev.benchmark as benchmark
    from kev.model import training_context
    max_state = int(pop_flag(argv, "--max_state"))
    QwenPredictor.task = pop_flag(argv, "--task")
    QwenPredictor.shots = int(pop_flag(argv, "--shots", "0"))
    benchmark.CONTEXT = {**training_context(max_state), "truncate": False}
    benchmark.LocalPredictor = QwenPredictor
    sys.argv = [sys.argv[0]] + argv
    benchmark.main()


if __name__ == "__main__":
    main(sys.argv[1:])
