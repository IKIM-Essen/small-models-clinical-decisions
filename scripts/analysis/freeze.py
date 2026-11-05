# /// script
# requires-python = ">=3.11"
# ///
"""Freeze each run's configuration before its one test pass: evaluate once on test, using the configurations
chosen on dev.

An entry pins, per row x task x seed: the run (or checkpoint) that is scored, the scorer (kev or qwen), the training
context, the dev scoring and the temperature kev.calibrate fitted on it, the test file and its sha256, where the test
scoring goes, and the commit. Entries are added group by group as rows finish on dev; an entry, once frozen, never
changes (a different value for an existing key is an error, not an update). configs/frozen.json is committed
before any test Job of its entries starts. Reads metrics and hashes files; never reads row content."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

TEST_DATA = {"p1": "splits/p1/test.jsonl", "p2": "data/derived/p2-test-nopairid.jsonl", "p3": "splits/p3/test.jsonl",
             "i1": "splits/i1/test.jsonl", "i2": "data/derived/i2-sf7400/test.novariant.jsonl", "i3": "splits/i3/test.jsonl"}
MAX_STATE = {"p1": 1024, "p2": 1024, "p3": 1024, "i1": 2048, "i2": 7552, "i3": 4096}
KEV4B_SNAPSHOT = os.path.expanduser("~/.cache/huggingface/hub/"
                                    "models--jaredpalmer--kev-4b/snapshots/"
                  "6cfce5c2fa4b4bd64026336ab649c5ca78857d52")
ROWS = ("kev4b-3k", "kev4b-full", "kev4b-zeroshot", "sft4b-3k", "qwen4b-zeroshot-k0", "qwen4b-zeroshot-k3", "classical-3k")
# classical baseline: trained on the same 3k subsets; I2's training split is its 3k subset
CLASSICAL_TRAIN = {"p1": "data/derived/p1-3k/train.jsonl", "p2": "data/derived/p2-3k/train.jsonl", "p3": "data/derived/p3-3k/train.jsonl",
                   "i1": "data/derived/i1-3k/train.jsonl", "i2": "data/derived/i2-sf7400/train.jsonl", "i3": "data/derived/i3-3k/train.jsonl"}
TASKS = tuple(TEST_DATA)


class NotReady(RuntimeError):
    pass


class FrozenChanged(RuntimeError):
    pass


class DuplicateTestOutput(RuntimeError):
    pass


def run_name(row: str, task: str, seed: int) -> str:
    if row == "kev4b-3k":   # I2's training split is 3,000 records: its full-data runs are its 3k runs
        return f"kev4b-{task}-full-s{seed}" if task == "i2" else f"kev4b-{task}-3k-s{seed}"
    if row == "kev4b-full":
        return f"kev4b-{task}-full-s{seed}"
    if row == "kev4b-zeroshot":
        return f"kev4b-{task}-zeroshot"
    if row == "sft4b-3k":
        return f"sft4b-{task}-3k-s{seed}"
    if row.startswith("qwen4b-zeroshot-k"):
        return f"qwen4b-{task}-zeroshot-k{row[-1]}"
    if row == "classical-3k":
        return f"classical-{task}-3k-s{seed}"
    raise ValueError(f"unknown row {row}")


def key(row: str, task: str, seed: int) -> str:
    return f"{row}/{task}/s{seed}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry(root: Path, row: str, task: str, seed: int, commit: str) -> dict:
    root = Path(root)
    run = f"runs/{run_name(row, task, seed)}"
    dev = root / f"{run}-dev"
    trained = row in ("kev4b-3k", "kev4b-full", "sft4b-3k")
    missing = [p for p in [dev / "report.json", dev / "calibration.json"] + ([root / run / "training_config.json"] if trained else [])
               if not p.exists()]
    if missing:
        raise NotReady(f"{key(row, task, seed)}: missing {', '.join(str(p.relative_to(root)) for p in missing)}")
    scorer = "qwen" if row.startswith(("sft4b", "qwen4b")) else "classical" if row == "classical-3k" else "kev"
    score_run = (KEV4B_SNAPSHOT if row == "kev4b-zeroshot" else "base" if row.startswith("qwen4b")
                 else CLASSICAL_TRAIN[task] if row == "classical-3k" else run)   # classical: the training subset it is refitted from
    shots = int(row[-1]) if row.startswith("qwen4b-zeroshot-k") else 0 if scorer == "qwen" else None
    test = TEST_DATA[task]
    return {"row": row, "task": task, "seed": seed, "run": run, "score_run": score_run, "scorer": scorer, "shots": shots,
            "max_state": 7552 if scorer == "classical" else MAX_STATE[task], "dev_dir": f"{run}-dev",
            "dev_temperature": float(json.loads((dev / "calibration.json").read_text())["workload_temperature"]),
            "dev_scored": json.loads((dev / "report.json").read_text())["coverage"]["evaluated_records"],
            "test_data": test, "test_data_sha256": _sha256(root / test), "test_out": f"{run}-test", "commit": commit}


def freeze(root: Path, out: Path, items, commit: str) -> dict:
    out = Path(out)
    frozen = json.loads(out.read_text()) if out.exists() else {
        "note": "Frozen configurations; one test pass per entry. Entries never change.", "entries": {}}
    for row, task, seed in items:
        e = entry(root, row, task, seed, commit)
        k = key(row, task, seed)
        if k in frozen["entries"]:
            old = {f: v for f, v in frozen["entries"][k].items() if f not in ("commit", "frozen_at")}
            if old != {f: v for f, v in e.items() if f != "commit"}:
                raise FrozenChanged(f"{k}: differs from its frozen entry")
            continue
        clash = [other for other, v in frozen["entries"].items() if v["test_out"] == e["test_out"]]
        if clash:   # e.g. kev4b-full/i2 is kev4b-3k/i2 (I2's full training split is its 3k subset): score it once
            raise DuplicateTestOutput(f"{k}: same test output as {clash[0]}")
        frozen["entries"][k] = {**e, "frozen_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(frozen, indent=1, sort_keys=True) + "\n")
    return frozen


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rows", required=True, help=f"comma list of {', '.join(ROWS)}")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="configs/frozen.json")
    ap.add_argument("--root", default=".")
    a = ap.parse_args(argv)
    root = Path(a.root)
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    items, skipped = [], []
    for row in a.rows.split(","):
        seeds = [0] if row in ("kev4b-zeroshot", "qwen4b-zeroshot-k0", "qwen4b-zeroshot-k3") else [int(s) for s in a.seeds.split(",")]
        for task in a.tasks.split(","):
            if row == "kev4b-full" and task == "i2":
                skipped.append("kev4b-full/i2: same runs as kev4b-3k/i2 (I2's training split is 3,000 records)")
                continue
            for seed in seeds:
                try:
                    entry(root, row, task, seed, commit)
                    items.append((row, task, seed))
                except NotReady as e:
                    skipped.append(str(e))
    frozen = freeze(root, Path(a.out), items, commit)
    print(f"{len(items)} entries frozen or confirmed, {len(frozen['entries'])} in {a.out}; {len(skipped)} not ready")
    for s in skipped:
        print("  not ready:", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
