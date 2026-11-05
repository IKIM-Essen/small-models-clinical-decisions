# /// script
# requires-python = ">=3.11"
# ///
"""Render the one test-scoring Job for frozen entries of configs/frozen.json.

The Job re-checks the test file's sha256 against the frozen entry, then scores the frozen run with the same scorer as on
dev (scripts/analysis/bench_ctx.py for Kev, scripts/qwen/bench_qwen.py for Qwen and SFT) into `<run>-test`. No training, no
calibration: the dev-fitted temperature is applied by scripts/analysis/stats.py. An entry that is not frozen cannot be rendered.

Usage: uv run --no-sync python scripts/analysis/testpass_job.py <row/task/sN> [...] | kubectl apply -f -"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

REPO = '"${KEV_ROOT:-$PWD}"'   # repository checkout inside the Job
DEADLINE_S = 86400


def score_command(e: dict) -> str:
    u = "uv run --no-sync python"
    if e["scorer"] == "kev":
        return (f"{u} scripts/analysis/bench_ctx.py --max_state {e['max_state']} --run {e['score_run']} --data {e['test_data']} "
                f"--device cuda --out {e['test_out']}")
    if e["scorer"] == "classical":   # refit on the frozen training subset with the frozen seed, reuse the dev-selected family/C
        return (f"{u} scripts/classical/bench_classical.py --train {e['score_run']} --task {e['task']} --seed {e['seed']} "
                f"--max_state {e['max_state']} --select {e['dev_dir']} --data {e['test_data']} --device cpu --out {e['test_out']}")
    return (f"{u} scripts/qwen/bench_qwen.py --max_state {e['max_state']} --task {e['task']} --shots {e['shots']} "
            f"--run {e['score_run']} --data {e['test_data']} --device cuda --out {e['test_out']}")


def render(frozen: dict, key: str, template: str, cpu_template: str | None = None) -> str:
    e = frozen["entries"][key]                     # KeyError: not frozen, not rendered
    if e["scorer"] == "classical" and cpu_template:   # no GPU: the CPU runner Job
        template = cpu_template
    name = "test-" + key.replace("/", "-")
    check = f"echo '{e['test_data_sha256']}  {e['test_data']}' | sha256sum -c --quiet"
    cmd = f"cd {REPO} && mkdir -p runs/logs && {{ {check} && {score_command(e)}; }} > runs/logs/{name}.log 2>&1"
    b64 = base64.b64encode(cmd.encode()).decode()
    return (template.replace("${RUNNER_NAME}", name).replace("${RUNNER_CMD_B64}", b64)
            .replace("${RUNNER_DEADLINE_S}", str(DEADLINE_S)))


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parents[2]
    frozen = json.loads((root / "configs" / "frozen.json").read_text())
    template = (root / "k8s" / "runner-job-gpu.yaml").read_text()
    cpu_template = (root / "k8s" / "runner-job.yaml").read_text()
    print("\n---\n".join(render(frozen, k, template, cpu_template) for k in argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
