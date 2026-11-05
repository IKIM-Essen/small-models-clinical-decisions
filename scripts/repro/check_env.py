"""Compare the live environment with uv.lock and configs/pins.json.

Run inside the project env: `uv run python scripts/repro/check_env.py` (no PEP 723
header on purpose: a script env would report every locked package as missing).

pins.json example: {"kev_commit": "569ea44", "vllm": "0.x.y"}
Exit 1 on drift. Also prints Python, CUDA and GPU so the run manifest can record them.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import platform
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from packaging.markers import Marker, default_environment


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _marker_true(marker: str | None, env: dict) -> bool:
    if not marker:
        return True
    return Marker(marker).evaluate(env)


def parse_lock(text: str, env: dict | None = None) -> dict[str, str]:
    """Packages the lock installs on this platform: walk the dependency edges from the project roots
    (editable / virtual sources), keeping only edges whose marker holds under `env` (default: the running
    interpreter). A multi-platform uv.lock also lists CUDA wheels and Windows-only packages; they are not
    drift on a Mac. A forked lock lists one package twice with different versions; an edge then carries the
    `version` it means. Editable packages (the project, third_party/kev) are pinned by commit, not version."""
    env = env or default_environment()
    lock = tomllib.loads(text)
    by_name: dict[str, list[dict]] = {}
    for pkg in lock.get("package", []):
        by_name.setdefault(_norm(pkg["name"]), []).append(pkg)

    def resolve(name: str, version: str | None) -> dict | None:
        cands = by_name.get(name, [])
        if version is not None:
            cands = [c for c in cands if c.get("version") == version]
        return cands[-1] if cands else None

    def is_root(pkg: dict) -> bool:
        src = pkg.get("source") or {}
        return "editable" in src or "virtual" in src

    roots = [(n, pkg) for n, pkgs in by_name.items() for pkg in pkgs if is_root(pkg)]
    seen: dict[tuple[str, str], set[str]] = {}   # (name, version) -> extras already expanded
    todo: list[tuple[str, str | None, list[str] | None]] = []

    def push_edges(edges: list[dict]) -> None:
        for edge in edges:
            if _marker_true(edge.get("marker"), env):
                todo.append((_norm(edge["name"]), edge.get("version"), edge.get("extra")))

    for _, pkg in roots:
        push_edges(pkg.get("dependencies", []))
        for group_edges in (pkg.get("dev-dependencies") or {}).values():
            push_edges(group_edges)
    while todo:
        name, version, extras = todo.pop()
        pkg = resolve(name, version)
        if pkg is None or is_root(pkg):
            continue
        key = (name, pkg["version"])
        wanted = set(extras or [])
        if key in seen and wanted <= seen[key]:
            continue
        first = key not in seen
        new_extras = wanted - seen.get(key, set())
        seen.setdefault(key, set()).update(wanted)
        edges = list(pkg.get("dependencies", [])) if first else []
        for extra in new_extras:
            edges += (pkg.get("optional-dependencies") or {}).get(extra, [])
        push_edges(edges)
    return {name: version for name, version in sorted(seen)}


def installed_packages() -> dict[str, str]:
    return {_norm(d.metadata["Name"]): d.version for d in md.distributions() if d.metadata["Name"]}


def compare(locked: dict[str, str], installed: dict[str, str]) -> list[dict]:
    locked = {_norm(k): v for k, v in locked.items()}
    installed = {_norm(k): v for k, v in installed.items()}
    drift = []
    for name, ver in sorted(locked.items()):
        if name not in installed:
            drift.append({"package": name, "problem": "missing", "locked": ver})
        elif installed[name] != ver:
            drift.append({"package": name, "problem": "version", "locked": ver, "installed": installed[name]})
    return drift


def check_pins(pins: dict, live: dict) -> list[dict]:
    problems = []
    for pin, want in pins.items():
        have = live.get(pin)
        if have is None:
            problems.append({"pin": pin, "problem": "unknown", "want": want})
        elif not (str(have).startswith(str(want)) or str(want).startswith(str(have))):
            problems.append({"pin": pin, "problem": "mismatch", "want": want, "have": have})
    return problems


def _git_head(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def live_pins(root: Path) -> dict:
    live = {}
    for key, rel in (("kev_commit", "third_party/kev"), ("laya_commit", "third_party/laya")):
        head = _git_head(root / rel)
        if head:
            live[key] = head
    try:
        live["vllm"] = md.version("vllm")
    except md.PackageNotFoundError:
        pass
    return live


def gpu_info() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "none"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path("."))
    a = ap.parse_args(argv)
    print(f"python {platform.python_version()} | {platform.platform()} | gpu: {gpu_info()}")
    problems = 0
    lock = a.root / "uv.lock"
    if lock.exists():
        for d in compare(parse_lock(lock.read_text()), installed_packages()):
            print(f"DRIFT {d}")
            problems += 1
    else:
        print("FAIL: uv.lock missing")
        problems += 1
    pins_file = a.root / "configs" / "pins.json"
    if pins_file.exists():
        for p in check_pins(json.loads(pins_file.read_text()), live_pins(a.root)):
            print(f"PIN {p}")
            problems += 1
    else:
        print("FAIL: configs/pins.json missing")
        problems += 1
    print("OK" if not problems else f"{problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
