"""Client for our OCR sidecar instance."""
from __future__ import annotations

import time

import httpx


def ocr_pdf(client: httpx.Client, base_url: str, pdf: bytes, retries: int = 3, backoff: float = 5.0, timeout: float = 600.0) -> str:
    last = None
    for attempt in range(retries + 1):
        r = client.post(f"{base_url.rstrip('/')}/ocr", content=pdf, headers={"content-type": "application/pdf"}, timeout=timeout)
        if r.status_code == 200:
            return r.json().get("text", "")
        last = r.status_code
        if r.status_code != 503:
            break
        time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"ocr sidecar returned HTTP {last}")
