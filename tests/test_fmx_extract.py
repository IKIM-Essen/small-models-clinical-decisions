import json

import httpx

from conftest import REPO, load_script
from kevmed.fmx import ocr

fx = load_script("scripts/data/fmx_extract.py")


def test_ocr_pdf_posts_binary_and_returns_text():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append((req.method, req.url.path, req.headers.get("content-type"), len(req.content)))
        return httpx.Response(200, json={"text": "SYNTHETIC markdown", "pages": [{"page": 1}], "total_regions": 1, "duration_s": 0.1})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert ocr.ocr_pdf(client, "http://ocr", b"%PDF-1.4 synthetic") == "SYNTHETIC markdown"
    assert calls == [("POST", "/ocr", "application/pdf", 18)]


def test_ocr_pdf_retries_503_then_raises():
    n = {"calls": 0}

    def handler(req):
        n["calls"] += 1
        return httpx.Response(503, json={"error": "queue full"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        ocr.ocr_pdf(client, "http://ocr", b"x", retries=2, backoff=0)
        assert False
    except RuntimeError as e:
        assert "503" in str(e) and n["calls"] == 3


def test_extract_writes_rows_and_counts_only(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("KEV_RESTRICTED_ROOT", str(tmp_path))
    rows = [{"report_id": "R1", "patient_id": "P1", "year": 2024, "issued": "2024-01-01", "icdo3_codes": ["8140/3~"], "markdown": "SYNTHETIC"}]
    n = fx.run_extract("i3", "2024-01-01", "2025-01-01", tmp_path, fetch_rows=lambda sql: iter(rows))
    assert n == {"rows.jsonl": 1}
    assert json.loads((tmp_path / "i3" / "rows.jsonl").read_text())["report_id"] == "R1"
    out = capsys.readouterr().out
    assert "SYNTHETIC" not in out and "1" in out


def test_i2_ocr_ledger_resumes(tmp_path, monkeypatch):
    monkeypatch.setenv("KEV_RESTRICTED_ROOT", str(tmp_path))
    rows = [{"case_id": "C1", "patient_id": "P1", "year": 2024, "drg_code": "I21.0", "icd_version": "2024", "doc_id": "D1",
             "doc_date": "2024-01-01", "pdf_url": "https://ship.example/app/docs/cla/1"}]
    served = {"n": 0}

    def handler(req):
        if req.url.host == "ship.example" and req.url.path == "/app/Auth/v1/basicAuth":
            assert req.method == "POST" and req.headers.get("authorization", "").startswith("Basic ")
            return httpx.Response(200, headers={"set-cookie": "shipToken=jwt-synthetic; Path=/; HttpOnly"})
        if req.url.host == "ship.example":
            assert req.headers.get("authorization") == "Bearer jwt-synthetic" and "cookie" not in req.headers   # SHIP's nginx rejects the 4 kB token as a cookie (400), accepts it as a bearer header
            return httpx.Response(200, content=b"%PDF-1.4 synthetic")
        served["n"] += 1
        return httpx.Response(200, json={"text": "SYNTHETIC md " * 20})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    fx.run_extract("i2", "2024-01-01", "2025-01-01", tmp_path, fetch_rows=lambda sql: iter(rows), ocr_url="http://ocr", client=client, ship_auth=("u", "p"), all_docs=True)
    assert (tmp_path / "i2" / "ocr" / "D1.md").exists() and served["n"] == 1
    fx.run_extract("i2", "2024-01-01", "2025-01-01", tmp_path, fetch_rows=lambda sql: iter(rows), ocr_url="http://ocr", client=client, ship_auth=("u", "p"), overwrite_rows=True, all_docs=True)
    assert served["n"] == 1                                                         # ledger skipped the finished doc


def test_ocr_requires_only_ids_unless_all(tmp_path, monkeypatch):
    monkeypatch.setenv("KEV_RESTRICTED_ROOT", str(tmp_path))
    rows = [{"case_id": f"C{i}", "patient_id": "P", "year": 2024, "drg_code": "I21.0", "icd_version": "2024", "doc_id": f"D{i}",
             "doc_date": "2024-01-01", "pdf_url": f"https://ship.example/app/docs/cla/{i}"} for i in range(3)]
    served = []

    def handler(req):
        if req.url.host == "ship.example" and req.url.path == "/app/Auth/v1/basicAuth":
            return httpx.Response(200, headers={"set-cookie": "shipToken=jwt-synthetic; Path=/"})
        if req.url.host == "ship.example":
            return httpx.Response(200, content=b"%PDF-1.4 synthetic")
        served.append(req)
        return httpx.Response(200, json={"text": "SYNTHETIC md " * 20})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    fx.run_extract("i2", "2024-01-01", "2025-01-01", tmp_path, fetch_rows=lambda sql: iter(rows))
    try:
        fx.run_ocr_step(tmp_path, ocr_url="http://ocr", client=client, ship_auth=("u", "p"))
        assert False
    except ValueError as e:
        assert "only-ids" in str(e)
    ids = tmp_path / "ids.txt"
    ids.write_text("D1\n")
    c = fx.run_ocr_step(tmp_path, ocr_url="http://ocr", client=client, ship_auth=("u", "p"), only_ids=ids)
    assert c["ok"] == 1 and len(served) == 1 and (tmp_path / "i2" / "ocr" / "D1.md").exists()
    assert not (tmp_path / "i2" / "ocr" / "D0.md").exists()
    c = fx.run_ocr_step(tmp_path, ocr_url="http://ocr", client=client, ship_auth=("u", "p"), all_docs=True)
    assert c["ok"] == 2 and c["skipped_done"] == 1


def test_extract_refuses_out_outside_restricted_root(tmp_path, monkeypatch):
    monkeypatch.setenv("KEV_RESTRICTED_ROOT", str(tmp_path / "restricted"))
    rows = [{"report_id": "R1", "patient_id": "P1", "year": 2024, "issued": "2024-01-01", "icdo3_codes": ["8140/3~"], "markdown": "SYNTHETIC"}]
    try:
        fx.run_extract("i3", "2024-01-01", "2025-01-01", tmp_path / "shared", fetch_rows=lambda sql: iter(rows))
        assert False
    except PermissionError as e:
        assert "restricted" in str(e) and "SYNTHETIC" not in str(e)
    n = fx.run_extract("i3", "2024-01-01", "2025-01-01", tmp_path / "restricted" / "internal" / "raw", fetch_rows=lambda sql: iter(rows))
    assert n == {"rows.jsonl": 1}


def test_ship_login_returns_cookie_and_fails_without_it():
    def ok(req):
        return httpx.Response(200, headers={"set-cookie": "shipToken=abc; Path=/"})
    assert fx.ship_login(httpx.Client(transport=httpx.MockTransport(ok)), "https://ship.example", ("u", "p")) == "abc"
    def bad(req):
        return httpx.Response(401)
    try:
        fx.ship_login(httpx.Client(transport=httpx.MockTransport(bad)), "https://ship.example", ("u", "p"))
        assert False
    except RuntimeError as e:
        assert "401" in str(e)


def test_month_chunks_cover_range_exactly():
    assert fx.month_chunks("2021-01-01", "2027-01-01", 12) == [(f"{y}-01-01", f"{y + 1}-01-01") for y in range(2021, 2027)]
    assert fx.month_chunks("2021-01-01", "2021-08-15", 3) == [("2021-01-01", "2021-04-01"), ("2021-04-01", "2021-07-01"), ("2021-07-01", "2021-08-15")]
    assert fx.month_chunks("2024-01-01", "2025-01-01", 0) == [("2024-01-01", "2025-01-01")]


def test_extract_chunked_runs_each_query_per_chunk_and_appends(tmp_path, monkeypatch):
    monkeypatch.setenv("KEV_RESTRICTED_ROOT", str(tmp_path))
    seen = []

    def fetch(sql):
        seen.append(sql)
        start = sorted(set(__import__("re").findall(r"'(\d{4}-\d{2}-\d{2})'", sql)))[0]
        return iter([{"report_id": f"R-{start}", "patient_id": "P1"}])

    n = fx.run_extract("i3", "2024-01-01", "2025-01-01", tmp_path, fetch_rows=fetch, chunk_months=6)
    assert n == {"rows.jsonl": 2}
    assert len(seen) == 2 and "'2024-07-01'" in seen[0] and "'2024-07-01'" in seen[1]
    ids = [json.loads(l)["report_id"] for l in (tmp_path / "i3" / "rows.jsonl").read_text().splitlines()]
    assert ids == ["R-2024-01-01", "R-2024-07-01"]


def test_extract_retries_a_cancelled_chunk_without_duplicating_rows(tmp_path, monkeypatch):
    import psycopg
    monkeypatch.setenv("KEV_RESTRICTED_ROOT", str(tmp_path))
    calls = {"n": 0}

    def fetch(sql):
        calls["n"] += 1
        start = sorted(set(__import__("re").findall(r"'(\d{4}-\d{2}-\d{2})'", sql)))[0]

        def gen():
            yield {"report_id": f"R-{start}-a", "patient_id": "P1"}
            if start == "2024-07-01" and calls["n"] == 2:            # first attempt at the second window dies mid-stream
                raise psycopg.errors.QueryCanceled("canceling statement due to user request")
            yield {"report_id": f"R-{start}-b", "patient_id": "P1"}
        return gen()

    n = fx.run_extract("i3", "2024-01-01", "2025-01-01", tmp_path, fetch_rows=fetch, chunk_months=6, retry_sleep=0)
    assert n == {"rows.jsonl": 4} and calls["n"] == 3
    ids = [json.loads(l)["report_id"] for l in (tmp_path / "i3" / "rows.jsonl").read_text().splitlines()]
    assert ids == ["R-2024-01-01-a", "R-2024-01-01-b", "R-2024-07-01-a", "R-2024-07-01-b"]
    assert not list((tmp_path / "i3").glob("*.part"))


def test_extract_gives_up_after_the_retries(tmp_path, monkeypatch):
    import psycopg
    monkeypatch.setenv("KEV_RESTRICTED_ROOT", str(tmp_path))

    def fetch(sql):
        raise psycopg.errors.QueryCanceled("canceling statement due to user request")

    try:
        fx.run_extract("i3", "2024-01-01", "2025-01-01", tmp_path, fetch_rows=fetch, retries=2, retry_sleep=0)
        assert False
    except psycopg.errors.QueryCanceled:
        pass


def test_ocr_runs_documents_concurrently_with_one_ledger_line_each(tmp_path, monkeypatch):
    import threading, time as _t
    monkeypatch.setenv("KEV_RESTRICTED_ROOT", str(tmp_path))
    rows = [{"doc_id": f"D{i}", "pdf_url": f"https://ship.example/app/docs/cla/{i}", "case_id": f"C{i}"} for i in range(12)]
    (tmp_path / "i2").mkdir()
    (tmp_path / "i2" / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    active, peak, lock = [0], [0], threading.Lock()

    def handler(req):
        if req.url.path.endswith("/basicAuth"):
            return httpx.Response(200, headers={"set-cookie": "shipToken=tok; Path=/"})
        if req.url.path == "/ocr":
            with lock:
                active[0] += 1; peak[0] = max(peak[0], active[0])
            _t.sleep(0.05)
            with lock:
                active[0] -= 1
            return httpx.Response(200, json={"text": "SYNTHETIC md"})
        return httpx.Response(200, content=b"%PDF synthetic")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    c = fx.run_ocr_step(tmp_path, ocr_url="http://ocr", client=client, ship_auth=("u", "p"), all_docs=True, workers=4)
    assert c["ok"] == 12 and c["failed"] == 0 and peak[0] > 1
    ledger = (tmp_path / "i2" / "ocr" / "ledger.jsonl").read_text().splitlines()
    assert len(ledger) == 12 and len({json.loads(l)["doc_id"] for l in ledger}) == 12
    assert sorted(p.name for p in (tmp_path / "i2" / "ocr").glob("D*.md")) == sorted(f"D{i}.md" for i in range(12))


def test_ocr_failures_record_a_masked_detail_and_wait_for_a_free_sidecar_slot(tmp_path, monkeypatch):
    monkeypatch.setenv("KEV_RESTRICTED_ROOT", str(tmp_path))
    rows = [{"doc_id": "D7", "pdf_url": "https://ship.example/app/docs/cla/12345678", "case_id": "C7"}]
    (tmp_path / "i2").mkdir()
    (tmp_path / "i2" / "rows.jsonl").write_text(json.dumps(rows[0]) + "\n")
    calls = {"ocr": 0}

    def handler(req):
        if req.url.path.endswith("/basicAuth"):
            return httpx.Response(200, headers={"set-cookie": "shipToken=tok; Path=/"})
        if req.url.path == "/ocr":
            calls["ocr"] += 1
            return httpx.Response(503, json={"error": "queue full"}) if calls["ocr"] < 5 else httpx.Response(200, json={"text": "SYNTHETIC"})
        return httpx.Response(200, content=b"%PDF synthetic")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    c = fx.run_ocr_step(tmp_path, ocr_url="http://ocr", client=client, ship_auth=("u", "p"), all_docs=True, workers=1, ocr_backoff=0)
    assert c["ok"] == 1 and calls["ocr"] == 5                                     # four 503s, then served

    def handler2(req):
        if req.url.path.endswith("/basicAuth"):
            return httpx.Response(200, headers={"set-cookie": "shipToken=tok; Path=/"})
        return httpx.Response(404, text="no document 12345678")
    (tmp_path / "i2" / "ocr" / "ledger.jsonl").unlink()
    client = httpx.Client(transport=httpx.MockTransport(handler2))
    c = fx.run_ocr_step(tmp_path, ocr_url="http://ocr", client=client, ship_auth=("u", "p"), all_docs=True, workers=1, ocr_backoff=0)
    led = json.loads((tmp_path / "i2" / "ocr" / "ledger.jsonl").read_text().splitlines()[0])
    assert led["status"] == "failed" and led["error"] == "HTTPStatusError" and "404" in led["detail"] and "12345678" not in led["detail"]


def test_ocr_refreshes_an_expired_ship_token_once(tmp_path, monkeypatch):
    """2026-09-28: the SHIP JWT expired after a few hours and 5,122 downloads failed with 401; a 401 on the document
    now triggers one fresh login for that origin and one retry."""
    monkeypatch.setenv("KEV_RESTRICTED_ROOT", str(tmp_path))
    rows = [{"doc_id": f"D{i}", "pdf_url": f"https://ship.example/app/docs/cla/{i}", "case_id": f"C{i}"} for i in range(3)]
    (tmp_path / "i2").mkdir()
    (tmp_path / "i2" / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    state = {"logins": 0}

    def handler(req):
        if req.url.path.endswith("/basicAuth"):
            state["logins"] += 1
            return httpx.Response(200, headers={"set-cookie": f"shipToken=tok{state['logins']}; Path=/"})
        if req.url.path == "/ocr":
            return httpx.Response(200, json={"text": "SYNTHETIC"})
        if req.headers.get("authorization") != "Bearer tok2" and req.url.path.endswith("/1"):
            return httpx.Response(401, text="expired")                              # token 1 expires before D1
        if req.headers.get("authorization") not in ("Bearer tok1", "Bearer tok2"):
            return httpx.Response(401, text="bad")
        return httpx.Response(200, content=b"%PDF synthetic")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    c = fx.run_ocr_step(tmp_path, ocr_url="http://ocr", client=client, ship_auth=("u", "p"), all_docs=True, workers=1, ocr_backoff=0)
    assert c["ok"] == 3 and c["failed"] == 0 and state["logins"] == 2
