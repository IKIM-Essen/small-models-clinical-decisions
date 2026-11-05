"""Build one task's splits from its source directory.
  uv run scripts/data/build_task.py --task p2 --src <source dir> --out splits/p2 --seed 20260925
Prints CONSORT counts and per-split label counts only. Never prints rows. Refuses to overwrite a split."""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
from pathlib import Path

MODULES = {"p1": "kevmed.tasks.p1_cxr", "p2": "kevmed.tasks.p2_mednli", "p3": "kevmed.tasks.p3_esi",
           "i1": "kevmed.tasks.i1_los", "i2": "kevmed.tasks.i2_letter", "i3": "kevmed.tasks.i3_patho"}


def _restricted_ok(task: str, out: Path) -> bool:
    from kevmed import splits
    return task not in splits.RESTRICTED_TASKS or splits.under_restricted_root(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", choices=sorted(MODULES), required=True)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--skip-bad", action="store_true", help="drop rows with an unmappable label instead of failing")
    ap.add_argument("--variant", default="main", help="builder variant where supported (i2: main | no_diagnosis)")
    ap.add_argument("--extracted-at", default=None, help="ISO timestamp of the FMX extraction (i1, i4: open cases and censoring)")
    a = ap.parse_args(argv)
    if not _restricted_ok(a.task, a.out):
        from kevmed import splits
        print(f"task {a.task} holds restricted rows: --out must be under $KEV_RESTRICTED_ROOT ({splits.restricted_root()})", file=sys.stderr)
        return 4
    from kevmed import splits
    mod = importlib.import_module(MODULES[a.task])
    try:
        kwargs = {"skip_bad": a.skip_bad}
        params = inspect.signature(mod.build).parameters
        if "extracted_at" in params:
            kwargs["extracted_at"] = a.extracted_at
        elif a.extracted_at:
            print(f"task {a.task} does not take --extracted-at", file=sys.stderr)
            return 2
        if "variant" in params:
            kwargs["variant"] = a.variant
        elif a.variant != "main":
            print(f"task {a.task} has no variants", file=sys.stderr)
            return 2
        by, consort = mod.build(a.src, a.seed, **kwargs)
        info = splits.write_splits(a.out, by, {"task": a.task, "kev_type": mod.KEV_TYPE, "seed": a.seed, "variant": a.variant,
                                               "extracted_at": a.extracted_at, "source": str(a.src), "consort": dict(consort)})
    except FileExistsError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 3
    print(json.dumps({"task": a.task, "consort": dict(consort),
                      "splits": {k: v["n"] for k, v in info["splits"].items()}}, indent=1))
    for name, s in info["splits"].items():
        for qid, q in s["questions"].items():
            print(f"  {name}/{qid} ({q['type']}): {q['labels']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
