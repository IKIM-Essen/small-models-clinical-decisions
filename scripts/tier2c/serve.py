"""Jev /v1/systemone server for the open System One models of tier 2c.

One process serves one checkpoint from the shared HF cache (pinned in configs/model_pins.json), so kev.benchmark
--remote scores it exactly like Kev, Jev and Laya (one eval client for every row):
    MODEL=clef-flash|clef|pplx-decider PORT=8030 uv run --no-sync python scripts/tier2c/serve.py
Clef and Clef-flash ship a `systemone()` that already speaks the wire format; pplx-decider ships a `DecisionModel`
that answers one (state, question) row at a time, wrapped here into the same {model, answers, usage} body.
Nothing is logged but counts; request bodies are never printed. Kev-27B needs no wrapper (kev.serve --run).
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

PINS = json.loads((Path(__file__).resolve().parents[2] / "configs" / "model_pins.json").read_text())["models"]
NAMES = {"clef-flash": "Cloudflare/clef-flash", "clef": "Cloudflare/clef", "pplx-decider": "perplexity-ai/pplx-decider-v1.1-27b"}


def load(name):
    repo = NAMES[name]
    path = Path(PINS[repo]["local_path"])
    if name.startswith("clef"):
        sys.path.insert(0, str(path))
        from joint_schema_model import load_release_model, systemone   # noqa: E402  (the checkpoint's own code)
        model, processor = load_release_model(path, device="cuda")

        def answer_request(body):
            return systemone(model, processor, {**body, "model": repo})
    else:
        sys.path.insert(0, str(path / "source" / "src"))
        from autojev.model import DecisionModel, answer   # noqa: E402
        model = DecisionModel(path, device="cuda")

        def answer_request(body):
            qids = list(body["questions"])
            rows = [{"state": body["state"], "question": body["questions"][q]} for q in qids]
            probs = model.predict(rows, batch_size=len(rows))
            return {"model": repo, "answers": {q: answer(body["questions"][q], p) for q, p in zip(qids, probs)},
                    "usage": {"input_tokens": 0, "output_tokens": 0}}
    return answer_request


def main():
    name = os.environ["MODEL"]
    port = int(os.environ.get("PORT", "8030"))
    answer_request = load(name)
    lock = threading.Lock()
    count = {"ok": 0, "err": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):   # quiet: no per-request lines (they would carry nothing useful and cost time)
            pass

        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code); self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data))); self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            self._send(200, {"status": "ok", "model": NAMES[name], "answered": count["ok"], "errors": count["err"]})

        def do_POST(self):
            if not self.path.endswith("/v1/systemone"):
                return self._send(404, {"error": "unknown route"})
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            try:
                with lock, torch.inference_mode():
                    out = answer_request(body)
                count["ok"] += 1
                self._send(200, out)
            except Exception as e:   # one bad record must not take the server down; kev.benchmark records it as rejected
                count["err"] += 1
                self._send(500, {"error": type(e).__name__})

    print(f"[tier2c] serving {NAMES[name]} on 127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
