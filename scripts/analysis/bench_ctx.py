# /// script
# requires-python = ">=3.11"
# ///
"""Kev.benchmark --data scored with the training context instead of kev's default 384-token context.

kev.benchmark has no flag for this: for --data it always uses kev.suite.CONTEXT (max_state 384) with
skip_overlong=True, so a model trained with --max_state 2048 has every longer record counted as rejected
(70% of I1 in the exploratory runs). This sets kev.benchmark.CONTEXT to
kev.model.training_context(--max_state) and runs the unchanged kev.benchmark main().

Usage: uv run python scripts/analysis/bench_ctx.py --max_state 2048 <kev.benchmark arguments>"""
import sys

import kev.benchmark as benchmark
from kev.model import training_context

i = sys.argv.index("--max_state")
max_state = int(sys.argv[i + 1])
del sys.argv[i:i + 2]
benchmark.CONTEXT = {**training_context(max_state), "truncate": False}
benchmark.main()
