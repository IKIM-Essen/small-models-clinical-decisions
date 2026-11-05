# /// script
# requires-python = ">=3.11"
# ///
"""Post hoc analysis: H2, H3 and H4b recomputed
exactly as scripts/analysis/hypotheses.py computes them, with every I3 test row restricted to the code-free, unseen-patient
test records (the records of runs/kev4b-i3-3k-nocode-s0-test, as in scripts/analysis/i3_nocode_rows.py). The models are
the original frozen ones; only the I3 record set changes. Output: runs/hypotheses-i3nocode-test.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import hypotheses as hy  # noqa: E402
import stats as st  # noqa: E402
from pertask_frontier import fast_bootstrap  # noqa: E402

KEYS: set = set()
_run_rows, _frontier_rows = hy.run_rows, hy.frontier_rows


def restrict(rows, task):
    return rows if task != "i3" or rows is None else [r for r in rows if st.key(r) in KEYS]


def run_rows(root, entry, split):
    return restrict(_run_rows(root, entry, split), entry["task"])


def frontier_rows(root, model, mode, task):
    return restrict(_frontier_rows(root, model, mode, task), task)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--samples", type=int, default=st.BOOTSTRAP_SAMPLES)
    ap.add_argument("--seed", type=int, default=st.BOOTSTRAP_SEED)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--only", default="H3,H4b,H2")
    ap.add_argument("--out", default="runs/hypotheses-i3nocode-test.json")
    a = ap.parse_args(argv)
    KEYS.update(st.key(r) for r in st.load_rows(ROOT / "runs/kev4b-i3-3k-nocode-s0-test"))
    hy.run_rows, hy.frontier_rows = run_rows, frontier_rows
    st.bootstrap = fast_bootstrap   # identical draws and rows, faster (tests/test_g4_optional.py)
    frozen = json.loads((ROOT / "configs" / "frozen.json").read_text())
    out_path = ROOT / a.out
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    out.update({"note": __doc__.split("\n\n")[0], "i3_decisions": len(KEYS), "i3_records": len({k[0] for k in KEYS}),
                "samples": a.samples, "seed": a.seed})
    compute = {"H3": lambda: hy.h3(ROOT, frozen, "test", hy.TASKS, a.samples, a.seed, a.jobs),
               "H4b": lambda: hy.h3(ROOT, frozen, "test", hy.INTERNAL, a.samples, a.seed, a.jobs),
               "H2": lambda: hy.h2(ROOT, frozen, a.samples, a.seed)}
    for h in a.only.split(","):
        out[h] = compute[h]()
        out_path.write_text(json.dumps(out, indent=1) + "\n")
        print(h, out[h].get("status"), out[h].get("passes"), flush=True)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
