# /// script
# requires-python = ">=3.11"
# ///
"""H1 latency part: p50 latency per decision.

Kev-4B zero shot (the pinned snapshot H1 names) and the LoRA-SFT 3k adapter (seed 0) are scored on the first 500 **dev**
records of each task by one Job each, pinned to one GPU product (A100-SXM4), fp32, batch 1, sequentially; both answer all
questions of a record in one forward pass, so per-decision latency = the record's latency_ms / its number of questions.
The frontier tier's H1 cells (`frontier`, non-thinking zero shot) come from the quiet-load measurement of 2026-10-26: one
request in flight per model, the same 500 dev records, per gateway call (`runs/frontier-ledgers/*-dev500.jsonl`). The
latencies seen during the 48 test runs (up to 160 requests in flight from this project) are kept as `frontier_under_load`
for the record and are not used for H1. Reads latency fields only."""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
from pathlib import Path

import numpy as np

TASKS = ("p1", "p2", "p3", "i1", "i2", "i3")
DEV = {"p1": "splits/p1/dev.jsonl", "p2": "data/derived/p2-dev-nopairid.jsonl", "p3": "splits/p3/dev.jsonl",
       "i1": "splits/i1/dev.jsonl", "i2": "data/derived/i2-sf7400/dev.novariant.jsonl", "i3": "splits/i3/dev.jsonl"}
MAX_STATE = {"p1": 1024, "p2": 1024, "p3": 1024, "i1": 2048, "i2": 7552, "i3": 4096}
SNAPSHOT = os.path.expanduser("~/.cache/huggingface/hub/models--jaredpalmer--kev-4b/snapshots/6cfce5c2fa4b4bd64026336ab649c5ca78857d52")
SFT_ADAPTER = "runs/sft4b-{t}-3k-s0"   # LoRA-SFT 3k, seed 0
PRODUCT = "NVIDIA-A100-SXM4-80GB"
N = 500


def _summary(values: list[float]) -> dict:
    v = np.asarray(values, dtype=float)
    return {"p50_ms": float(np.median(v)), "p95_ms": float(np.quantile(v, 0.95))}


def kev_per_decision(predictions: Path) -> dict:
    per = []
    for line in Path(predictions).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            per.append(d["prediction"]["latency_ms"] / max(1, len(d["rows"])))
    return {**_summary(per), "n_records": len(per), "per_decision_ms": per}


def frontier_per_call(ledgers: list[Path]) -> dict:
    ms = []
    for f in ledgers:
        for line in Path(f).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("status") == "ok" and r.get("latency_s") is not None:
                    ms.append(1000 * r["latency_s"])
    return {**_summary(ms), "n_calls": len(ms)}


def speedup(frontier: dict, kev: dict) -> float:
    return frontier["p50_ms"] / kev["p50_ms"]


def render_job(template: str, name: str = "latency-kev4b-zeroshot-dev500") -> str:
    steps = ["mkdir -p data/derived/latency runs/logs"]
    for t in TASKS:
        sub = f"data/derived/latency/{t}-dev500.jsonl"
        steps.append(f"head -n {N} {DEV[t]} > {sub}")
        steps.append(f"uv run --no-sync python scripts/analysis/bench_ctx.py --max_state {MAX_STATE[t]} --run {SNAPSHOT} "
                     f"--data {sub} --device cuda --out runs/latency-kev4b-zeroshot-{t}-dev500")
    cmd = "cd \"${KEV_ROOT:-$PWD}\" && { " + " && ".join(steps) + f"; }} > runs/logs/{name}.log 2>&1"
    y = (template.replace("${RUNNER_NAME}", name).replace("${RUNNER_CMD_B64}", base64.b64encode(cmd.encode()).decode())
         .replace("${RUNNER_DEADLINE_S}", "259200"))
    return y.replace("values: [NVIDIA-A100-SXM4-80GB, NVIDIA-A100-80GB-PCIe]", f"values: [{PRODUCT}]")


def render_sft_job(template: str, name: str = "latency-sft4b-3k-dev500") -> str:
    """The LoRA-SFT counterpart of render_job: same 500 dev records, same GPU product, scripts/qwen/bench_qwen.py."""
    steps = ["mkdir -p data/derived/latency runs/logs"]
    for t in TASKS:
        sub = f"data/derived/latency/{t}-dev500.jsonl"
        steps.append(f"[ -s {sub} ] || head -n {N} {DEV[t]} > {sub}")
        steps.append(f"uv run --no-sync python scripts/qwen/bench_qwen.py --max_state {MAX_STATE[t]} --task {t} "
                     f"--run {SFT_ADAPTER.format(t=t)} --data {sub} --device cuda --out runs/latency-sft4b-3k-{t}-dev500")
    cmd = "cd \"${KEV_ROOT:-$PWD}\" && { " + " && ".join(steps) + f"; }} > runs/logs/{name}.log 2>&1"
    y = (template.replace("${RUNNER_NAME}", name).replace("${RUNNER_CMD_B64}", base64.b64encode(cmd.encode()).decode())
         .replace("${RUNNER_DEADLINE_S}", "172800"))
    return y.replace("values: [NVIDIA-A100-SXM4-80GB, NVIDIA-A100-80GB-PCIe]", f"values: [{PRODUCT}]")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--render", action="store_true", help="print the pinned Kev latency Job")
    ap.add_argument("--render-sft", action="store_true", help="print the pinned LoRA-SFT latency Job")
    ap.add_argument("--out", default="runs/latency.json")
    a = ap.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if a.render or a.render_sft:
        print((render_sft_job if a.render_sft else render_job)((root / "k8s" / "runner-job-gpu.yaml").read_text()))
        return 0
    out = {"note": __doc__.split("\n\n")[0], "gpu_product": PRODUCT,
           "measurement": {"kev": "A100-SXM4, fp32, batch 1, 500 dev records per task",
                           "sft": "LoRA-SFT 3k seed 0, same records and conditions as Kev",
                           "frontier": "quiet gateway load, one request in flight per model, 2026-10-26 08:30-10:50 Berlin, same 500 dev records",
                           "frontier_under_load": "per call from the 48 test-run ledgers, up to 160 requests in flight from this project"},
           "kev": {}, "sft": {}, "frontier": {}, "frontier_under_load": {}, "speedup_p50": {}, "speedup_under_load_p50": {}, "speedup_sft_p50": {}}
    for t in TASKS:
        p = root / f"runs/latency-kev4b-zeroshot-{t}-dev500/predictions.jsonl"
        if p.exists():
            k = kev_per_decision(p); k.pop("per_decision_ms")
            out["kev"][t] = k
        p = root / f"runs/latency-sft4b-3k-{t}-dev500/predictions.jsonl"
        if p.exists():
            k = kev_per_decision(p); k.pop("per_decision_ms")
            out["sft"][t] = k
            if t in out["kev"]:
                out["speedup_sft_p50"][t] = speedup(k, out["kev"][t])
    for model in ("deepseek-v4.1-flash", "kimi-k3"):
        for t in TASKS:
            quiet = root / f"runs/frontier-ledgers/frontier-{model}-nothink-k0-{t}-dev500.jsonl"
            if quiet.exists():
                fr = frontier_per_call([quiet])
                out["frontier"][f"{model}/nothink-k0/{t}"] = fr
                if t in out["kev"]:
                    out["speedup_p50"][f"{model}/nothink-k0/{t}"] = speedup(fr, out["kev"][t])
        for mode in ("nothink-k0", "nothink-k3", "think-k0", "think-k3"):
            for t in TASKS:
                ledgers = [Path(f) for f in glob.glob(str(root / f"runs/frontier-ledgers/frontier-{model}-{mode}-{t}-test.jsonl"))]
                if ledgers:
                    fr = frontier_per_call(ledgers)
                    out["frontier_under_load"][f"{model}/{mode}/{t}"] = fr
                    if t in out["kev"]:
                        out["speedup_under_load_p50"][f"{model}/{mode}/{t}"] = speedup(fr, out["kev"][t])
    (root / a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"kev tasks {len(out['kev'])}, sft tasks {len(out['sft'])}, frontier quiet cells {len(out['frontier'])}, "
          f"under load {len(out['frontier_under_load'])} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
