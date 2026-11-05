"""Runner-pod extraction: runs the extract SQL for one task and writes raw JSONL under
data/internal/raw/<task>/. Prints row counts only. Runs inside the hospital network only.
  uv run scripts/data/fmx_extract.py --task i3 --start 2019-01-01 --end 2027-01-01 --out data/internal/raw
  uv run scripts/data/fmx_extract.py --task i2 --start 2023-01-01 --end 2027-01-01 --out data/internal/raw \
      --ocr http://ocr-sidecar:8090 --ship-user-env SHIP_USER --ship-pass-env SHIP_PASS
FMX_DSN, SHIP_USER, SHIP_PASS come from the runner's Kubernetes secret."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator

import httpx

from kevmed import splits
from kevmed.fmx import ocr, sql

QUERIES = {
    "i1": {"rows.jsonl": sql.extract_i1, "labs.jsonl": sql.extract_i1_labs, "history.jsonl": sql.extract_i1_history},
    "i2": {"rows.jsonl": sql.extract_i2},
    "i3": {"rows.jsonl": sql.extract_i3},
    "i4": {"rows.jsonl": sql.extract_i4},
}


def _default(o):
    return o.isoformat() if hasattr(o, "isoformat") else str(o)


def month_chunks(start: str, end: str, months: int) -> list[tuple[str, str]]:
    """Half-open [start, end) split into consecutive `months`-long windows aligned to `start` (last one ends at `end`).
    `months <= 0` means one window. Keeps every FMX statement short enough for the server-side cancel that killed the
    26-minute I1 labs query (`QueryCanceled: canceling statement due to user request`, statement_timeout is 0)."""
    from datetime import date
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    if months <= 0 or s >= e:
        return [(start, end)]
    out, cur = [], s
    while cur < e:
        m = cur.month - 1 + months
        nxt = min(date(cur.year + m // 12, m % 12 + 1, cur.day), e)
        out.append((cur.isoformat(), nxt.isoformat()))
        cur = nxt
    return out


def _write(path: Path, rows: Iterable[dict], append: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a" if append else "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_default) + "\n")
            n += 1
    return n


def _write_window(target: Path, fetch_rows: Callable[[str], Iterator[dict]], statement: str, append: bool,
                  retries: int, retry_sleep: float) -> int:
    """One window into a `.part` file, appended to `target` only when the statement completed: FMX cancels
    statements from outside the session (`QueryCanceled`), and a half-written window must not leak into the output.
    Retries the same window up to `retries` times (transient cancels and connection drops), then re-raises."""
    import psycopg
    part = target.with_suffix(target.suffix + ".part")
    for attempt in range(retries + 1):
        try:
            n = _write(part, fetch_rows(statement))
            break
        except (psycopg.errors.QueryCanceled, psycopg.OperationalError) as e:
            print(json.dumps({"retry": attempt + 1, "error": type(e).__name__, "target": target.name}), file=sys.stderr)
            if attempt == retries:
                part.unlink(missing_ok=True)
                raise
            time.sleep(retry_sleep)
    with target.open("a" if append else "w", encoding="utf-8") as out, part.open(encoding="utf-8") as src:
        for line in src:
            out.write(line)
    part.unlink()
    return n


def _ledger_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("status") == "ok":
                done.add(rec["doc_id"])
    return done


def ship_login(client: httpx.Client, base_url: str, auth: tuple[str, str]) -> str:
    """SHIP issues a JWT as a `shipToken` cookie on POST /app/Auth/v1/basicAuth with HTTP basic credentials
    (the same flow MeDict's backend uses); document downloads carry it as a bearer header, not the credentials."""
    r = client.post(f"{base_url.rstrip('/')}/app/Auth/v1/basicAuth", auth=auth, timeout=30.0)
    if r.status_code != 200:
        raise RuntimeError(f"SHIP login failed with HTTP {r.status_code}")
    token = r.cookies.get("shipToken")
    if not token:
        for header in r.headers.get_list("set-cookie"):
            if header.startswith("shipToken="):
                token = header.split(";", 1)[0].split("=", 1)[1]
    if not token:
        raise RuntimeError("SHIP login returned no shipToken cookie")
    return token


_DIGITS = re.compile(r"\d{5,}")


def run_ocr(rows_path: Path, out_dir: Path, ocr_url: str, client: httpx.Client, ship_auth: tuple[str, str],
            only_ids: set[str] | None = None, workers: int = 4, ocr_timeout: float = 1800.0, ocr_retries: int = 20,
            ocr_backoff: float = 1.0) -> dict:
    """OCR the selected letters through the sidecar with `workers` documents in flight (the sidecar queues
    MAX_CONCURRENT_DOCS per replica and answers 503 beyond that, which `ocr.ocr_pdf` retries with a short backoff:
    the Service picks replicas at random, a 10 s backoff left replicas idle while workers slept, 158 letters/h). One SHIP login per
    document origin, shared cookie; ledger lines are written under a lock, one per finished document."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = out_dir / "ledger.jsonl"
    done = _ledger_done(ledger)
    counts = {"docs": 0, "skipped_not_selected": 0, "skipped_done": 0, "ok": 0, "failed": 0}
    todo = []
    with rows_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            counts["docs"] += 1
            if only_ids is not None and r["doc_id"] not in only_ids:
                counts["skipped_not_selected"] += 1
                continue
            if r["doc_id"] in done:
                counts["skipped_done"] += 1
                continue
            todo.append(r)
    token_by_origin: dict[str, str] = {}
    lock = threading.Lock()

    def refresh(url: str, stale: str) -> None:
        """Replace an expired token (SHIP JWTs expire after a few hours; 5,122 downloads failed with 401 on
        2026-09-28). Only the first worker that sees the stale token logs in again."""
        u = httpx.URL(url)
        origin = f"{u.scheme}://{u.host}" + (f":{u.port}" if u.port else "")
        with lock:
            if token_by_origin.get(origin) == stale:
                token_by_origin[origin] = ship_login(client, origin, ship_auth)
                client.cookies.clear()

    def headers_for(url: str) -> dict[str, str]:
        """One login per document origin; the token travels as `Authorization: Bearer` (SHIP's nginx answers
        400 "Request Header Or Cookie Too Large" when the 4 kB JWT is sent as the shipToken cookie, 2026-09-28)."""
        u = httpx.URL(url)
        origin = f"{u.scheme}://{u.host}" + (f":{u.port}" if u.port else "")
        with lock:
            if origin not in token_by_origin:
                token_by_origin[origin] = ship_login(client, origin, ship_auth)
                client.cookies.clear()                                       # the jar would resend the oversized cookie
            return {"Authorization": f"Bearer {token_by_origin[origin]}"}

    def one(r: dict) -> dict:
        t0 = time.time()
        try:
            h = headers_for(r["pdf_url"])
            pdf = client.get(r["pdf_url"], headers=h, timeout=120.0)
            if pdf.status_code == 401:                                       # expired token: one fresh login, one retry
                refresh(r["pdf_url"], h["Authorization"].removeprefix("Bearer "))
                pdf = client.get(r["pdf_url"], headers=headers_for(r["pdf_url"]), timeout=120.0)
            pdf.raise_for_status()
            text = ocr.ocr_pdf(client, ocr_url, pdf.content, retries=ocr_retries, backoff=ocr_backoff, timeout=ocr_timeout)
            (out_dir / f"{r['doc_id']}.md").write_text(text, encoding="utf-8")
            return {"doc_id": r["doc_id"], "status": "ok", "chars": len(text), "seconds": round(time.time() - t0, 1)}
        except Exception as e:  # noqa: BLE001  (error class and a masked message, never content)
            return {"doc_id": r["doc_id"], "status": "failed", "error": type(e).__name__,
                    "detail": _DIGITS.sub("<n>", str(e))[:120], "seconds": round(time.time() - t0, 1)}

    with ledger.open("a", encoding="utf-8") as led, ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for rec in pool.map(one, todo):
            with lock:
                led.write(json.dumps(rec) + "\n")
                led.flush()
                counts["ok" if rec["status"] == "ok" else "failed"] += 1
                n = counts["ok"] + counts["failed"]
                if n % 50 == 0:
                    print(json.dumps({"ocr_progress": n, "of": len(todo), "ok": counts["ok"], "failed": counts["failed"]}), file=sys.stderr, flush=True)
    return counts


def run_ocr_step(out: Path, ocr_url: str, client: httpx.Client | None = None, ship_auth: tuple[str, str] | None = None,
                 only_ids: Path | None = None, all_docs: bool = False, workers: int = 4, ocr_backoff: float = 1.0) -> dict:
    """OCR the I2 letters listed in <out>/i2/rows.jsonl. Refuses to OCR every letter (tens of thousands) unless
    `all_docs`; otherwise `only_ids` (one doc_id per line, e.g. the sampled cases) is required."""
    if only_ids is None and not all_docs:
        raise ValueError("OCR needs --only-ids <file> (one doc_id per line) or --all")
    ids = {l.strip() for l in Path(only_ids).read_text().splitlines() if l.strip()} if only_ids else None
    splits.require_restricted(Path(out), "fmx_extract --out (OCR)")
    task_dir = Path(out) / "i2"
    c = run_ocr(task_dir / "rows.jsonl", task_dir / "ocr", ocr_url, client or httpx.Client(), ship_auth or ("", ""), ids, workers=workers, ocr_backoff=ocr_backoff)
    print(json.dumps({"ocr": c}))
    return c


def run_extract(task: str, start: str, end: str, out: Path, fetch_rows: Callable[[str], Iterator[dict]] | None = None,
                ocr_url: str | None = None, client: httpx.Client | None = None, ship_auth: tuple[str, str] | None = None,
                overwrite_rows: bool = False, only_ids: Path | None = None, all_docs: bool = False,
                chunk_months: int = 0, retries: int = 2, retry_sleep: float = 60.0) -> dict:
    if fetch_rows is None:
        from kevmed.fmx import db
        fetch_rows = db.rows
    splits.require_restricted(Path(out), f"fmx_extract --out for {task}")
    task_dir = Path(out) / task
    counts = {}
    for fname, q in QUERIES[task].items():
        target = task_dir / fname
        if target.exists() and not overwrite_rows:
            raise FileExistsError(f"{target} exists (raw extracts are written once; pass --overwrite to redo)")
        counts[fname] = 0
        for i, (a, b) in enumerate(month_chunks(start, end, chunk_months)):
            counts[fname] += _write_window(target, fetch_rows, q(a, b), append=i > 0, retries=retries, retry_sleep=retry_sleep)
            print(json.dumps({"task": task, "file": fname, "window": [a, b], "rows_so_far": counts[fname]}), file=sys.stderr)
    print(json.dumps({"task": task, "start": start, "end": end, "chunk_months": chunk_months, "rows": counts}))
    if task == "i2" and ocr_url:
        counts["ocr"] = run_ocr_step(out, ocr_url, client, ship_auth, only_ids, all_docs)
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", choices=sorted(QUERIES), required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ocr", default=None, help="OCR sidecar base URL (i2 only)")
    ap.add_argument("--ship-user-env", default="SHIP_USER")
    ap.add_argument("--ship-pass-env", default="SHIP_PASS")
    ap.add_argument("--overwrite", action="store_true", help="re-run the extract query over an existing raw file")
    ap.add_argument("--ocr-only", action="store_true", help="i2: skip extraction, OCR the existing rows.jsonl")
    ap.add_argument("--only-ids", type=Path, default=None, help="i2 OCR: file with one doc_id per line (the sampled letters)")
    ap.add_argument("--all", action="store_true", help="i2 OCR: every letter in rows.jsonl (tens of thousands; deliberate)")
    ap.add_argument("--ocr-workers", type=int, default=4, help="documents in flight against the sidecar (MAX_CONCURRENT_DOCS x replicas)")
    ap.add_argument("--chunk-months", type=int, default=0, help="run each extract query per N-month window of admission dates (0 = one statement)")
    a = ap.parse_args(argv)
    auth = (os.environ.get(a.ship_user_env, ""), os.environ.get(a.ship_pass_env, ""))
    if a.ocr_only:
        if a.task != "i2" or not a.ocr:
            print("--ocr-only needs --task i2 and --ocr <url>", file=sys.stderr)
            return 2
        run_ocr_step(a.out, a.ocr, ship_auth=auth, only_ids=a.only_ids, all_docs=a.all, workers=a.ocr_workers)
        return 0
    run_extract(a.task, a.start, a.end, a.out, ocr_url=a.ocr, ship_auth=auth, overwrite_rows=a.overwrite, only_ids=a.only_ids, all_docs=a.all,
                chunk_months=a.chunk_months)
    return 0


if __name__ == "__main__":
    sys.exit(main())
