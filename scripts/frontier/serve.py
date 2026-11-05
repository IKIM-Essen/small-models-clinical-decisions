"""/v1/systemone adapter for the frontier tier.

Turns one System One request (state + typed questions) into one gateway call per question for `deepseek-v4.1-flash` or
`kimi-k3`, so `kev.benchmark --remote` scores the frontier rows exactly like Kev, the open System One models and Laya.

    FRONTIER_MODEL=deepseek-v4.1-flash FRONTIER_MODE=nothink|think FRONTIER_SHOTS=0|3 FRONTIER_TASK=p2 \
    FRONTIER_LEDGER=runs/.../ledger.jsonl PORT=8040 uv run --no-sync python scripts/frontier/serve.py

Prompt rules (identical for every generative model): fixed system prompt; options as a lettered list (noul as A/B,
score levels as letters too); option order shuffled per example with a seed derived from the state and question id;
answer with the letter only; temperature 0. Non-thinking: thinking switched off through
`chat_template_kwargs={"thinking": false}` (verified on both deployments 2026-10-16), max_tokens 8, probability =
softmax over the first-token logprobs of the option letters. Thinking: label only (the letter after the trace), which
is returned as a one-hot distribution; thinking rows are reported with accuracy and macro-F1 only. Few shot: k=3
records from the task's train split (seed 20261016), the same for every model and mode, shown as user/assistant turns
with their own shuffled options. Every call goes through scripts/llm/litellm_client.py (budget guard, retry-after,
fallback discard on a wrong backend) and is appended to a JSONL ledger; a restarted run answers finished calls from it.
A call served by a fallback backend, or failing with 5xx/429 after the client's retries, is retried every minute for up
to FRONTIER_WAIT_MIN (default 180) so a backend restart does not end the run; a parse failure gets a uniform
distribution and is counted in frontier_counts.json (a guided rerun is added if a task's rate exceeds 1%).
Never logs prompts, states or keys.
"""
import hashlib
import json
import os
import random
import string
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "llm"))
import litellm_client as lc  # noqa: E402
import token_guard as tg  # noqa: E402

BASE_URL = os.environ.get("LLM_GATEWAY_URL", "http://localhost:4000/v1")   # OpenAI-compatible LiteLLM gateway
# Backend(s) each model must be served by, as JSON {"model": "url" or ["url", ...]}; a call answered by any other
# backend went through a gateway fallback and is discarded. Every ledger record keeps `served_by`.
EXPECTED_API_BASE = json.loads(os.environ.get("LLM_EXPECTED_API_BASE", "{}"))
SYSTEM = ("You are a careful clinical decision assistant. You read one document and answer one multiple-choice "
          "question about it. Answer with the letter of the correct option only.")
FEWSHOT_SEED = 20261016
LETTERS = string.ascii_uppercase


def options(q):
    """(keys, descriptions) in canonical order; keys are what kev expects back."""
    t = q["type"]
    if t == "noul":
        return ["true", "false"], ["yes", "no"]
    crit = q.get("criteria")
    if t == "score":
        return [str(i) for i in range(len(crit))], [str(c) for c in crit]
    if isinstance(crit, dict):
        return [str(k) for k in crit], [f"{k}: {v}" if v else str(k) for k, v in crit.items()]
    return [str(c) for c in crit], [str(c) for c in crit]


def shuffled(state, qid, q):
    keys, desc = options(q)
    order = list(range(len(keys)))
    seed = int(hashlib.sha256(f"{qid}\x00{state}".encode()).hexdigest()[:12], 16)
    random.Random(seed).shuffle(order)
    return [keys[i] for i in order], [desc[i] for i in order]


def user_turn(state, q, desc):
    lines = [f"{LETTERS[i]}) {d}" for i, d in enumerate(desc)]
    return f"Document:\n{state}\n\nQuestion: {q['instructions']}\nOptions:\n" + "\n".join(lines) + \
        "\n\nAnswer with the option letter only."


def load_fewshot(task, k=3):
    """K train records per task, fixed seed; their questions carry the labels used as the assistant answers."""
    path = ROOT / "splits" / task / "train.jsonl"
    if task == "i2":
        path = ROOT / "data" / "derived" / "i2-sf7400" / "train.jsonl"
    with open(path) as f:
        n = sum(1 for _ in f)
    pick = sorted(random.Random(FEWSHOT_SEED).sample(range(n), k))
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i in pick:
                out.append(json.loads(line))
    return out


def label_key(q):
    t, y = q["type"], q["label"]
    if t == "noul":
        return "true" if y else "false"
    return str(y)


class Frontier:
    WAIT_STATUSES = ("fallback_discarded", "server", "rate_limit")
    def __init__(self):
        self.model = os.environ["FRONTIER_MODEL"]
        self.think = os.environ.get("FRONTIER_MODE", "nothink") == "think"
        self.shots = int(os.environ.get("FRONTIER_SHOTS", "0"))
        self.task = os.environ["FRONTIER_TASK"]
        self.trace_cap = int(os.environ.get("FRONTIER_TRACE_CAP", "4096"))
        # definitions arm: the task's label conventions and training-label frequencies go into the
        # system prompt; everything else (examples, shuffling, parsing) stays identical
        self.defs = os.environ.get("FRONTIER_DEFS", "0") == "1"
        self.system = SYSTEM
        if self.defs:
            block = json.loads((ROOT / "scripts" / "frontier" / "label_definitions.json").read_text())[self.task]["text"]
            self.system = SYSTEM + "\n\nLabel conventions for this task: " + block
        self.wait_min = int(os.environ.get("FRONTIER_WAIT_MIN", "180"))   # how long to wait out a backend outage per call
        self.client = lc.LiteLLMClient(BASE_URL, dict(os.environ), expected_api_base={self.model: EXPECTED_API_BASE[self.model]} if self.model in EXPECTED_API_BASE else None)
        self.fewshot = load_fewshot(self.task) if self.shots else []
        self.ledger = Path(os.environ["FRONTIER_LEDGER"])
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.done = {}
        if self.ledger.exists():
            for line in self.ledger.read_text().splitlines():
                r = json.loads(line)
                if r.get("status") == "ok":
                    self.done[r["key"]] = r
        self.lock = threading.Lock()
        self.counts = {"calls": 0, "cached": 0, "parse_failure": 0, "fallback_discarded": 0, "other_error": 0}

    def messages(self, state, qid, q):
        msgs = [{"role": "system", "content": getattr(self, "system", SYSTEM)}]   # qwen_lib calls this on a bare namespace
        for ex in self.fewshot:
            eq = ex["questions"].get(qid)
            if eq is None:           # few-shot record without this question (multi-question tasks): skip it
                continue
            k, d = shuffled(ex["state"], qid, eq)
            msgs += [{"role": "user", "content": user_turn(ex["state"], eq, d)},
                     {"role": "assistant", "content": LETTERS[k.index(label_key(eq))]}]
        keys, desc = shuffled(state, qid, q)
        msgs.append({"role": "user", "content": user_turn(state, q, desc)})
        return msgs, keys

    def ask(self, state, qid, q):
        msgs, keys = self.messages(state, qid, q)
        letters = list(LETTERS[:len(keys)])
        payload = {"model": self.model, "messages": msgs, "temperature": 0,
                   "chat_template_kwargs": {"thinking": self.think}}
        if not self.think:
            payload.update(logprobs=True, top_logprobs=20, max_tokens=tg.NON_THINKING_MAX_TOKENS)
        key = tg.request_key(payload)
        with self.lock:
            hit = self.done.get(key)
        if hit is not None:
            self.counts["cached"] += 1
            rec = hit
        else:
            deadline = time.time() + 60 * self.wait_min
            while True:
                if self.think:
                    rec = self.client.chat(payload, thinking=True, trace_cap=self.trace_cap, letters_answer_only=letters)
                else:
                    rec = self.client.chat(payload, letters=letters)
                rec = {**rec, "key": key, "task": self.task, "qid": qid, "letters": letters}
                with self.lock:
                    self.counts["calls"] += 1
                    tg.ledger_append(self.ledger, rec)
                    if rec.get("status") == "ok":
                        self.done[key] = rec
                # backend outage (gateway served a fallback model, or 5xx after the client's own retries): wait it out
                # instead of failing the record; kev.benchmark aborts the whole run after 3 failed attempts per record
                if rec.get("status") in self.WAIT_STATUSES and time.time() < deadline:
                    self.counts[rec["status"] if rec["status"] in self.counts else "other_error"] += 1
                    time.sleep(60)
                    continue
                break
        status = rec.get("status")
        if status == "parse_failure":
            # reported per run (frontier_counts.json); the record gets a uniform distribution so the run continues and the
            # miss counts against the model. If the rate is above 1% on a task, a guided label-only rerun is added.
            self.counts["parse_failure"] += 1
            keys = self.messages(state, qid, q)[1]
            return {k: 1.0 / len(keys) for k in keys}, rec
        if status != "ok":
            self.counts[status if status in self.counts else "other_error"] += 1
            raise RuntimeError(status)
        if self.think:
            probs = {k: (1.0 if LETTERS[i] == rec["answer"] else 0.0) for i, k in enumerate(keys)}
        else:
            probs = {k: float(rec["probs"].get(LETTERS[i], 0.0)) for i, k in enumerate(keys)}
            z = sum(probs.values())
            probs = {k: v / z for k, v in probs.items()}
        return probs, rec

    def answer(self, body):
        answers, latency, usage_in, usage_out = {}, 0.0, 0, 0
        for qid, q in body["questions"].items():
            probs, rec = self.ask(body["state"], qid, q)
            latency += rec.get("latency_s") or 0.0
            u = rec.get("usage") or {}
            usage_in += u.get("prompt_tokens") or 0
            usage_out += u.get("completion_tokens") or 0
            if q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": probs["true"]}
            else:
                best = max(probs, key=probs.get)
                answers[qid] = {"type": q["type"], "probabilities": probs, "confidence": probs[best],
                                ("score" if q["type"] == "score" else "choice"): (int(best) if q["type"] == "score" else best)}
        return {"model": self.model, "answers": answers,
                "usage": {"input_tokens": usage_in, "output_tokens": usage_out, "gateway_latency_s": latency}}


def main():
    fr = Frontier()
    port = int(os.environ.get("PORT", "8040"))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, {"status": "ok", "model": fr.model, "think": fr.think, "shots": fr.shots, "defs": fr.defs, **fr.counts})

        def do_POST(self):
            if not self.path.endswith("/v1/systemone"):
                return self._send(404, {"error": "unknown route"})
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            try:
                self._send(200, fr.answer(body))
            except Exception as e:   # one bad record must not stop the run; kev.benchmark retries 5xx, then rejects
                self._send(503, {"error": str(e)[:80]})

    print(f"[frontier] {fr.model} think={fr.think} shots={fr.shots} task={fr.task} on 127.0.0.1:{port}, "
          f"{len(fr.done)} cached calls", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
