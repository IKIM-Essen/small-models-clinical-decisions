"""Seeded, group-aware, label-stratified splits and immutable split files.
Aggregates only: build.json holds counts, hashes and provenance, never rows."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from .schema import summarize, write_jsonl

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESTRICTED_ROOT = str(REPO_ROOT)   # row files live in the repo checkout, never in git
RESTRICTED_TASKS = {"p1", "p2", "p3", "s1", "s2", "i1", "i2", "i3", "i4"}   # row files must stay on the restricted volume


def restricted_root() -> Path:
    return Path(os.environ.get("KEV_RESTRICTED_ROOT", DEFAULT_RESTRICTED_ROOT)).resolve()


def under_restricted_root(path: Path) -> bool:
    """True when `path` (resolved, symlinks followed) is the restricted root or inside it."""
    root = restricted_root()
    target = Path(path).resolve()
    return target == root or root in target.parents


def require_restricted(path: Path, what: str) -> None:
    if not under_restricted_root(path):
        raise PermissionError(f"{what} holds row-level data: the target must be under $KEV_RESTRICTED_ROOT ({restricted_root()})")


def group_split(groups: dict[str, str], labels: dict[str, str], fractions: dict[str, float], seed: int) -> dict[str, str]:
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    members: dict[str, list[str]] = defaultdict(list)
    for rid, g in groups.items():
        members[g].append(rid)
    strata: dict[str, list[str]] = defaultdict(list)
    for g, ids in members.items():
        majority = Counter(labels[i] for i in ids).most_common(1)[0][0]
        strata[str(majority)].append(g)
    rng = random.Random(seed)
    names = list(fractions)
    out: dict[str, str] = {}
    for stratum in sorted(strata):
        gs = sorted(strata[stratum])
        rng.shuffle(gs)
        n = len(gs)
        bounds, acc = [], 0.0
        for name in names:
            acc += fractions[name]
            bounds.append(round(acc * n))
        start = 0
        for name, end in zip(names, bounds):
            for g in gs[start:end]:
                for rid in members[g]:
                    out[rid] = name
            start = end
    return out


def assign(records: list[dict], split_of: dict[str, str]) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by[split_of[r["_meta"]["id"]]].append(r)
    return dict(by)


def overlap(by_split: dict[str, list[dict]], key: str) -> int:
    seen: dict[str, set[str]] = defaultdict(set)
    for name, recs in by_split.items():
        for r in recs:
            seen[str(r["_meta"][key])].add(name)
    return sum(1 for s in seen.values() if len(s) > 1)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def write_splits(out_dir: Path, by_split: dict[str, list[dict]], build_info: dict) -> dict:
    out_dir = Path(out_dir)
    for name in by_split:
        if (out_dir / f"{name}.jsonl").exists():
            raise FileExistsError(f"{out_dir / (name + '.jsonl')} exists: splits are immutable once written")
    out_dir.mkdir(parents=True, exist_ok=True)
    info = dict(build_info)
    info["git_commit"] = _git_commit()
    info["splits"], info["files"] = {}, {}
    for name, recs in by_split.items():
        p = out_dir / f"{name}.jsonl"
        n = write_jsonl(p, recs)
        info["splits"][name] = summarize(recs)
        info["files"][p.name] = {"records": n, "sha256": _sha256(p), "bytes": p.stat().st_size}
    (out_dir / "build.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    return info
