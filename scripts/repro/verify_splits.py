# /// script
# requires-python = ">=3.11"
# ///
"""Verify frozen splits against splits/CHECKSUMS.sha256.

Prints file names and hashes only, never file content. Exit 1 on any problem.
  uv run scripts/repro/verify_splits.py --splits splits            # verify
  uv run scripts/repro/verify_splits.py --splits splits --write    # freeze (once, when the splits are built)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

SPLIT_SUFFIXES = {".jsonl", ".csv", ".parquet"}   # row files only; build.json / token_stats.json live in git
CHECKSUM_NAME = "CHECKSUMS.sha256"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


NOT_SPLITS = {"review"}   # splits/review/: label-check answers, they grow while people review (not frozen)


def _split_files(splits_dir: Path) -> list[str]:
    return sorted(
        p.relative_to(splits_dir).as_posix()
        for p in splits_dir.rglob("*")
        if p.is_file() and p.suffix in SPLIT_SUFFIXES and p.relative_to(splits_dir).parts[0] not in NOT_SPLITS
    )


def parse_checksums(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            digest, name = line.split(maxsplit=1)
            out[name.strip().lstrip("*")] = digest
    return out


def verify(splits_dir: Path, checksum_file: Path) -> list[dict]:
    listed = parse_checksums(checksum_file)
    problems = []
    for name, digest in sorted(listed.items()):
        p = splits_dir / name
        if not p.exists():
            problems.append({"file": name, "problem": "missing"})
        elif sha256(p) != digest:
            problems.append({"file": name, "problem": "mismatch"})
    for name in _split_files(splits_dir):
        if name not in listed:
            problems.append({"file": name, "problem": "unlisted"})
    return problems


def write(splits_dir: Path, checksum_file: Path, force: bool = False) -> tuple[int, list[str]]:
    """Freeze: append lines for new files, keep existing lines. Returns (lines written, files whose hash changed).
    A changed hash is refused (nothing written) unless `force`: frozen splits are immutable (Section 6.c)."""
    existing = parse_checksums(checksum_file) if checksum_file.exists() else {}
    existing = {n: h for n, h in existing.items() if Path(n).suffix in SPLIT_SUFFIXES and Path(n).parts[0] not in NOT_SPLITS}   # drop lines for files no longer tracked
    current = {n: sha256(splits_dir / n) for n in _split_files(splits_dir)}
    changed = sorted(n for n, h in current.items() if n in existing and existing[n] != h)
    if changed and not force:
        return 0, changed
    merged = {**existing, **current}
    lines = [f"{h}  {n}" for n, h in sorted(merged.items())]
    checksum_file.parent.mkdir(parents=True, exist_ok=True)
    checksum_file.write_text("\n".join(lines) + "\n")
    return len(lines), changed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--splits", type=Path, default=Path("splits"))
    ap.add_argument("--checksums-file", type=Path, default=None, help=f"default: <splits>/{CHECKSUM_NAME}")
    ap.add_argument("--write", action="store_true", help="freeze: add checksums for new split files (existing lines are kept)")
    ap.add_argument("--force", action="store_true", help="with --write: allow an existing hash to change")
    a = ap.parse_args(argv)
    checksum_file = a.checksums_file or (a.splits / CHECKSUM_NAME)
    if a.write:
        n, changed = write(a.splits, checksum_file, a.force)
        if changed and not a.force:
            for name in changed:
                print(f"REFUSED {name}: hash changed (frozen split; use --force only with a recorded decision)")
            return 3
        print(f"wrote {n} checksums to {checksum_file}" + (f" ({len(changed)} changed with --force)" if changed else ""))
        return 0
    if not checksum_file.exists():
        print(f"FAIL: {checksum_file} not found", file=sys.stderr)
        return 1
    problems = verify(a.splits, checksum_file)
    for p in problems:
        print(f"FAIL {p['problem']}: {p['file']}")
    if not problems:
        print(f"OK: {len(parse_checksums(checksum_file))} split files match {checksum_file}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
