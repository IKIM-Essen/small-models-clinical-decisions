"""Kev.benchmark --remote with a long per-request timeout for the frontier adapter (scripts/frontier/serve.py).

kev's RemotePredictor waits 120 s per record and aborts the whole run after 3 failed attempts. A frontier record can
legitimately take longer: P1 asks 14 questions per report (14 sequential gateway calls), thinking traces run up to
4,096 tokens, and the adapter deliberately waits out a backend restart (FRONTIER_WAIT_MIN). Retried records are cheap:
every finished gateway call is answered from the adapter's ledger. Everything else is the unchanged kev.benchmark.

Usage: uv run --no-sync python scripts/frontier/bench_remote.py <kev.benchmark --remote arguments>"""
import kev.benchmark as benchmark
import kev.predictors as predictors

_init = predictors.RemotePredictor.__init__


def _patched(self, base_url, model="kev-latest", api_key="local", timeout=4 * 3600, retries=5, concurrency=1):
    _init(self, base_url, model, api_key, timeout=timeout, retries=retries, concurrency=concurrency)


predictors.RemotePredictor.__init__ = _patched
benchmark.RemotePredictor = predictors.RemotePredictor
benchmark.main()
