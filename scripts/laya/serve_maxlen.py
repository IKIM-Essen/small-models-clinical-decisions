"""Laya-serve with a default per-request max_len ().

kev.benchmark --remote sends the Jev /v1/systemone body without `max_len`, so laya-serve would
fall back to each checkpoint's shipped limit (multilingual: 1,024 tokens) and silently cut long
German states (I1 p50 is about 800 tokens). Laya's README asks for `max_len=8192` on long
documents; this wrapper adds that default to every /v1/systemone(/batch) body that does not set
it, and otherwise serves laya.serve.create_app() unchanged. Same env vars as laya-serve, plus
LAYA_DEFAULT_MAX_LEN (default 8192). Never logs request bodies.
"""
import json
import os

import uvicorn
from laya.serve import create_app

DEFAULT_MAX_LEN = int(os.environ.get("LAYA_DEFAULT_MAX_LEN", "8192"))


class DefaultMaxLen:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or not scope["path"].endswith(("/v1/systemone", "/v1/systemone/batch")):
            return await self.app(scope, receive, send)
        chunks, more = [], True
        while more:
            msg = await receive()
            chunks.append(msg.get("body", b""))
            more = msg.get("more_body", False)
        body = b"".join(chunks)
        try:
            data = json.loads(body)
            if isinstance(data, dict) and "max_len" not in data:
                data["max_len"] = DEFAULT_MAX_LEN
                body = json.dumps(data).encode()
        except ValueError:
            pass
        headers = [(k, v) for k, v in scope["headers"] if k != b"content-length"] + [(b"content-length", str(len(body)).encode())]
        sent = False

        async def replay():
            nonlocal sent
            if sent:
                return await receive()
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        return await self.app({**scope, "headers": headers}, replay, send)


if __name__ == "__main__":
    uvicorn.run(DefaultMaxLen(create_app()), host=os.environ.get("LAYA_HOST", "127.0.0.1"),
                port=int(os.environ.get("LAYA_PORT", "8020")), log_level=os.environ.get("LAYA_LOG_LEVEL", "warning"))
