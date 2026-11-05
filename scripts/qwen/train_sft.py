# /// script
# requires-python = ">=3.11"
# ///
"""LoRA-SFT baseline for H3/H4b: Qwen3.5-4B Instruct fine-tuned to answer each question with its option letter
.

One training example per question of every record of the given file (the same 3k subset file Kev trains on), prompt
built by scripts/qwen/qwen_lib.py (= the frontier prompt), loss = cross-entropy of the answer letter at the answer
position only. Fixed recipe for every task: LoRA rank 16, alpha 32, dropout 0.05 on all linear layers (Kev-4B's
adapter settings), AdamW lr 1e-4, weight decay 0.01, linear warmup over 3% of the steps then cosine, batch 8 x accumulation 1
(Kev's batch; right padding, which cannot change a causal model's real positions, see qwen_lib.answer_logits),
gradient clipping 1.0, bf16 weights, gradient checkpointing. Records whose document exceeds --max_state are dropped
and counted, as kev.train does. Refuses an existing --out. Prints counts and losses only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_lib as ql  # noqa: E402

RECIPE = {"lora": 16, "lora_alpha": 32, "lora_dropout": 0.05, "lora_targets": "all-linear", "lr": 1e-4,
          "weight_decay": 0.01, "warmup_frac": 0.03, "schedule": "cosine", "batch": 8, "accum": 1,
          "padding": "right", "grad_clip": 1.0, "dtype": "bf16", "gradient_checkpointing": True}


def batches(order: list[int], size: int) -> list[list[int]]:
    return [order[i:i + size] for i in range(0, len(order), size)]


def build_examples(tok, records: list[dict], max_state: int) -> tuple[list[tuple[list[int], int]], int]:
    examples, dropped = [], 0
    for r in records:
        ex = ql.training_examples(tok, r, max_state)
        if ex:
            examples.extend(ex)
        else:
            dropped += 1
    return examples, dropped


def epoch_order(n: int, seed: int, epoch: int) -> list[int]:
    order = list(range(n))
    random.Random(f"{seed}:{epoch}").shuffle(order)
    return order


def n_optimizer_steps(n_examples: int, epochs: int, batch: int) -> int:
    return math.ceil(n_examples / batch) * epochs


def subset_mismatch(data: Path, subsets_dir: Path) -> str | None:
    """If `data` is a 3k training subset (listed as `out` in a runs/subsets summary), its sha256 must match the summary:
    Kev's and SFT's 3k rows must train on byte-identical files (H3 "identical data"). Returns a reason, or None."""
    for f in sorted(Path(subsets_dir).glob("*.json")):
        summary = json.loads(f.read_text())
        if Path(summary.get("out", "")).resolve() == Path(data).resolve():
            got = _sha256(Path(data))
            return None if got == summary["out_sha256"] else f"{data} sha256 {got[:12]} != {f.name} {summary['out_sha256'][:12]}"
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_state", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_steps", type=int, default=0, help="stop after this many optimizer steps (timing runs only)")
    ap.add_argument("--subsets_dir", default=str(ql.ROOT / "runs" / "subsets"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        print(f"{out} exists: refusing to overwrite")
        return 1
    reason = subset_mismatch(Path(a.data), Path(a.subsets_dir))
    if reason:
        print(f"refusing to train: {reason}")
        return 1

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(ql.BASE, revision=ql.BASE_REVISION)
    records = [json.loads(line) for line in open(a.data, encoding="utf-8") if line.strip()]
    examples, dropped = build_examples(tok, records, a.max_state)
    print(f"{len(records)} records, {dropped} dropped over {a.max_state} tokens, {len(examples)} examples", flush=True)

    model = AutoModelForCausalLM.from_pretrained(ql.BASE, revision=ql.BASE_REVISION, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=RECIPE["lora"], lora_alpha=RECIPE["lora_alpha"],
                                             lora_dropout=RECIPE["lora_dropout"], target_modules="all-linear",
                                             task_type="CAUSAL_LM"))
    model.to(a.device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=RECIPE["lr"], weight_decay=RECIPE["weight_decay"])
    total = n_optimizer_steps(len(examples), a.epochs, RECIPE["batch"])
    if a.max_steps:
        total = min(total, a.max_steps)
    sched = get_cosine_schedule_with_warmup(opt, max(1, round(RECIPE["warmup_frac"] * total)), total)
    print(f"trainable params {sum(p.numel() for p in params) / 1e6:.1f}M, {total} optimizer steps", flush=True)

    causal = model.get_base_model()
    pad_id = tok.pad_token_id or 0
    start, step, seen, run_loss, step_seconds = time.time(), 0, 0, 0.0, []
    for epoch in range(a.epochs):
        for chunk in batches(epoch_order(len(examples), a.seed, epoch), RECIPE["batch"]):
            t_step = time.time()
            z = ql.answer_logits(causal, [examples[i][0] for i in chunk], a.device, pad_id)
            y = torch.tensor([examples[i][1] for i in chunk], device=a.device)
            loss = torch.nn.functional.cross_entropy(z.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, RECIPE["grad_clip"])
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            seen += len(chunk)
            run_loss += loss.item() * len(chunk)
            step_seconds.append(time.time() - t_step)
            if step % 10 == 0 or step == total:
                print(f"ep{epoch} step {step}/{total} loss {run_loss / seen:.3f}", flush=True)
            if step >= total:
                break
        if step >= total:
            break

    out.mkdir(parents=True)
    model.save_pretrained(out)
    (out / "training_config.json").write_text(json.dumps({
        "args": vars(a), "base": ql.BASE, "base_revision": ql.BASE_REVISION, "recipe": RECIPE,
        "data_sha256": _sha256(Path(a.data)), "n_records": len(records), "dropped_records": dropped,
        "n_examples": len(examples)}, indent=1) + "\n")
    (out / "training_metrics.json").write_text(json.dumps({
        "wall_seconds": time.time() - start, "optimizer_steps": step, "examples_seen": seen,
        "mean_train_loss": run_loss / max(1, seen), "step_seconds": step_seconds,
        "peak_device_bytes": torch.cuda.max_memory_allocated() if a.device == "cuda" else None}, indent=1) + "\n")
    print(f"saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
