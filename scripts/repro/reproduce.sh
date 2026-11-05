#!/usr/bin/env bash
# Rebuild runs/results/metrics.json from the aggregate run outputs in runs/ and check it against the committed copy.
# Needs only uv: the script uses the standard library. Re-running the experiments themselves needs the data and GPUs
# (see README).
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "== results: runs/ -> runs/results/metrics.json"
uv run --no-project python scripts/analysis/results_macros.py

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if git diff --quiet -- runs/results/metrics.json; then
    echo "== runs/results/metrics.json matches the committed copy"
  else
    echo "FAIL: runs/results/metrics.json differs from the committed copy" >&2
    git --no-pager diff --stat -- runs/results/metrics.json >&2
    exit 1
  fi
fi
echo "reproduce: OK"
