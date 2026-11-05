# /// script
# requires-python = ">=3.11"
# ///
"""P50/p95 latency per decision from run reports, for the descriptive latency table (H1 latency part).
Kev rows are local per-record scoring on whichever A100 the Job got; remote rows (gateway, served models) are
end-to-end under the recorded concurrency. Not comparable as measured: H1's number comes from the pinned
measurement. Reads report.json only."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def collect(report_paths: list[Path]) -> list[dict]:
    rows = []
    for p in sorted(report_paths):
        r = json.loads(Path(p).read_text())
        lat = r.get("latency_ms")
        if not lat:
            continue
        remote = r.get("remote")
        rows.append({"run": Path(p).parent.name, "split": r.get("split"), "p50_ms": lat.get("median"),
                     "p95_ms": lat.get("p95"), "n": (r.get("coverage") or {}).get("evaluated_records"),
                     "remote": bool(remote), "concurrency": (remote or {}).get("concurrency")})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--glob", default="runs/*/report.json")
    ap.add_argument("--out", default="runs/latency-dev.json")
    a = ap.parse_args(argv)
    rows = collect([Path(p) for p in glob.glob(a.glob)])
    Path(a.out).write_text(json.dumps({"note": __doc__.split("\n\n")[0], "rows": rows}, indent=1) + "\n")
    print(f"{len(rows)} runs with latency -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
