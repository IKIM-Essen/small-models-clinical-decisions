"""Write splits/<task>/token_stats.json: state token histograms per split file (aggregates only).
  uv run scripts/data/token_stats.py --split splits/s3 [--base Qwen/Qwen3.5-4B-Base]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kevmed import tokens


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--base", default=tokens.DEFAULT_BASE)
    a = ap.parse_args(argv)
    tok = tokens.load_tokenizer(a.base)
    out = {"base": a.base, "files": {}}
    for p in sorted(a.split.glob("*.jsonl")):
        s = tokens.split_stats(p, tok)
        out["files"][p.name] = s
        h = s["state"]
        print(f"{p.name}: n={h['n']} p50={h['p50']} p95={h['p95']} max={h['max']} over_384={h['over_384']} over_1024={h['over_1024']} qmax={s['question_max']}")
    (a.split / "token_stats.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
