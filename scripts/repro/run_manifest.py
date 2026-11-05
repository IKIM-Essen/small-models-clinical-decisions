# /// script
# requires-python = ">=3.11"
# ///
"""Write runs/<id>/manifest.json.

  uv run scripts/repro/run_manifest.py --run-id kev08b-transfer-v4-dev \
      --model jaredpalmer/kev-0.8b --config '{"suite": "evals/v4/transfer-v4"}' \
      --split-hash <sha256> --seed 0 --deployment-tag mlx-bf16 --start <iso> --end <iso>

Records git commit and dirty flag, host, GPU, Python. Never prints or stores rows.
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def gpu_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout.strip()
        return out or "none"
    except (OSError, subprocess.CalledProcessError):
        pass
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        return f"apple-silicon:{chip or 'unknown'}"
    return "none"


def build_manifest(run_id: str, model: str, config: dict, split_hashes: list[str], seed: int | None,
                   start: str | None, end: str | None, deployment_tag: str, root: Path) -> dict:
    return {
        "run_id": run_id,
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_dirty": bool(_git(root, "status", "--porcelain")),
        "config": config,
        "split_hashes": split_hashes,
        "model": model,
        "deployment_tag": deployment_tag,
        "seed": seed,
        "start": start,
        "end": end,
        "host": socket.gethostname(),
        "gpu": gpu_name(),
        "python": platform.python_version(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default="{}", help="JSON object")
    ap.add_argument("--split-hash", action="append", default=[])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--deployment-tag", default="")
    ap.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    ap.add_argument("--root", type=Path, default=REPO_ROOT, help="git repo the run traces to (default: this repo)")
    a = ap.parse_args(argv)
    try:
        config = json.loads(a.config)
    except json.JSONDecodeError as e:
        print(f"--config is not valid JSON: {e}", file=sys.stderr)
        return 2
    if not isinstance(config, dict):
        print("--config must be a JSON object", file=sys.stderr)
        return 2
    m = build_manifest(a.run_id, a.model, config, a.split_hash, a.seed, a.start, a.end, a.deployment_tag, a.root)
    if not m["git_commit"]:
        print(f"no git commit found at --root {a.root}: a run must trace to a commit (Section 6.a)", file=sys.stderr)
        return 3
    out = a.runs_dir / a.run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    print(f"wrote {out / 'manifest.json'} (commit {m['git_commit'][:7]}{' dirty' if m['git_dirty'] else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
