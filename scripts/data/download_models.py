# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub"]
# ///
"""Revision-pinned downloader for HF model weights.

This pins whole model
snapshots by Hub commit. First download of a model resolves and freezes the current HEAD
revision into --pins (default configs/model_pins.json, key "models"); every later run of this script
re-downloads exactly that pinned revision, never a newer HEAD, until a human deletes the pin.

Registry: the open System One models (zero-shot tier 2c) plus Kev-27B.

Usage:
  uv run scripts/data/download_models.py --only clef-flash
  uv run scripts/data/download_models.py --only clef,pplx-decider,kev-27b
  uv run scripts/data/download_models.py --verify

This script never prints model weight content, only repo ids, revisions, and byte counts.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_PINS = REPO_ROOT / "configs" / "model_pins.json"

REGISTRY = {
    "clef-flash": "Cloudflare/clef-flash",
    "clef": "Cloudflare/clef",
    "pplx-decider": "perplexity-ai/pplx-decider-v1.1-27b",
    # Superseded by v1.1 in tier 2c; kept so --verify still covers its pin.
    "pplx-decider-v1": "perplexity-ai/pplx-decider-v1-27b",
    "kev-27b": "jaredpalmer/kev-27b",
}


def today() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def human(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.1f}{unit}"
        f /= 1024
    return f"{f:.1f}TB"


def dir_size_bytes(path: Path) -> int:
    # Excludes __pycache__: importing a model's custom modeling code (e.g. a smoke check) writes
    # a .pyc next to it, which is incidental to running the code, not part of the downloaded asset.
    return sum(p.stat().st_size for p in path.rglob("*")
               if p.is_file() and "__pycache__" not in p.parts)


def resolve_revision(api, repo_id: str, pins: dict) -> str:
    """Return the pinned revision if one exists; otherwise resolve and freeze the current HEAD."""
    pinned = pins.get(repo_id, {}).get("revision")
    if pinned:
        return pinned
    return api.model_info(repo_id).sha


def download_one(repo_id: str, revision: str, dest: Path, snapshot_fn=snapshot_download) -> str:
    dest.mkdir(parents=True, exist_ok=True)
    return snapshot_fn(repo_id, revision=revision, cache_dir=str(dest))


def default_dest() -> Path:
    import os

    env = os.environ.get("HF_HOME")
    return Path(env).expanduser() if env else Path.home() / ".cache" / "huggingface"


def load_pins(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def write_pins(path: Path, pins: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(pins, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def verify(pins: dict, only: set[str] | None) -> list[str]:
    problems = []
    models = pins.get("models", {})
    names = only or set(REGISTRY.values())
    checked_any = False
    for repo_id in sorted(names):
        entry = models.get(repo_id)
        if entry is None:
            problems.append(f"NOPIN    {repo_id}: not in pins")
            continue
        checked_any = True
        local_path = entry.get("local_path")
        if not local_path or not Path(local_path).is_dir():
            problems.append(f"MISSING  {repo_id}: {local_path}")
            continue
        got = dir_size_bytes(Path(local_path))
        if got != entry.get("local_size_bytes"):
            problems.append(f"SIZE     {repo_id}: on disk {got} != pinned {entry.get('local_size_bytes')}")
    if not checked_any and not problems:
        problems.append("NOFILES  nothing to check")
    return problems


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--only", default=None, help=f"comma-separated subset of: {', '.join(REGISTRY)}")
    p.add_argument("--dest", type=Path, default=None, help="cache_dir for snapshot_download (default: $HF_HOME)")
    p.add_argument("--pins", type=Path, default=DEFAULT_PINS)
    p.add_argument("--verify", action="store_true", help="no download; check pinned models are present at their recorded size")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pins = load_pins(args.pins)

    if args.only:
        repo_ids = [REGISTRY[k] for k in args.only.split(",")]
    else:
        repo_ids = list(REGISTRY.values())

    if args.verify:
        problems = verify(pins, set(repo_ids) if args.only else None)
        for pr in problems:
            print(pr)
        print(f"[verify] {'FAIL' if problems else 'OK'}")
        return 1 if problems else 0

    dest = (args.dest or default_dest()).expanduser().resolve()
    api = HfApi()
    models = pins.setdefault("models", {})
    for repo_id in repo_ids:
        revision = resolve_revision(api, repo_id, models)
        print(f"[{repo_id}] revision={revision}")
        local_path = download_one(repo_id, revision, dest, snapshot_fn=snapshot_download)
        size = dir_size_bytes(Path(local_path))
        print(f"  {local_path} ({human(size)})")
        models[repo_id] = {
            "revision": revision,
            "resolved_at": models.get(repo_id, {}).get("resolved_at") or today(),
            "local_path": str(local_path),
            "local_size_bytes": size,
        }
    write_pins(args.pins, pins)
    print(f"[pins] {args.pins}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
