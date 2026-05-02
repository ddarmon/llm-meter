"""FastAPI app: reverse-proxies an OpenAI/Anthropic-compatible LLM endpoint
and serves a single-page dashboard that meters token usage in real time.

Captures full request bodies (messages + system + tools) and reconstructs
response content (including tool calls) from streaming events, so the UI
can show the exact transcript the agent sent and highlight which messages
are new this turn vs. replayed from earlier turns.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)

STATIC_DIR = Path(__file__).parent / "static"

# Comparison-pricing table, USD per million tokens.
# Keys are stable IDs surfaced to the frontend; `label` is for display.
PRICING: dict[str, dict[str, Any]] = {
    "opus_47": {
        "label": "Claude Opus 4.7",
        "input": 5.00,
        "output": 25.00,
        "cache_write_5m": 6.25,  # 1.25x input
        "cache_read": 0.50,  # 0.10x input
    },
    "sonnet_46": {
        "label": "Claude Sonnet 4.6",
        "input": 3.00,
        "output": 15.00,
        "cache_write_5m": 3.75,
        "cache_read": 0.30,
    },
    "haiku_45": {
        "label": "Claude Haiku 4.5",
        "input": 1.00,
        "output": 5.00,
        "cache_write_5m": 1.25,
        "cache_read": 0.10,
    },
}


# ---------- usage ------------------------------------------------------------


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    thinking_tokens: int = 0

    def merge(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_creation_tokens += other.cache_creation_tokens
        self.thinking_tokens += other.thinking_tokens

    def is_empty(self) -> bool:
        return (
            self.input_tokens == 0
            and self.output_tokens == 0
            and self.cache_read_tokens == 0
            and self.cache_creation_tokens == 0
            and self.thinking_tokens == 0
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "thinking_tokens": self.thinking_tokens,
        }


def _cost(u: Usage, p: dict[str, Any]) -> float:
    return (
        u.input_tokens * p["input"]
        + u.output_tokens * p["output"]
        + u.cache_read_tokens * p["cache_read"]
        + u.cache_creation_tokens * p["cache_write_5m"]
    ) / 1_000_000


def _costs_for(u: Usage) -> dict[str, float]:
    return {mid: _cost(u, p) for mid, p in PRICING.items()}


def compute_multipliers() -> list[dict[str, Any]]:
    """Compute pairwise cost multipliers between every pair of models.

    Returns a list of {from_model, from_label, to_model, to_label,
    input_mult, output_mult} for every ordered pair where from > to.
    """
    mids = list(PRICING.keys())
    result: list[dict[str, Any]] = []
    for i, hi in enumerate(mids):
        for lo in mids[i + 1 :]:
            p_hi = PRICING[hi]
            p_lo = PRICING[lo]
            input_mult = p_hi["input"] / p_lo["input"]
            output_mult = p_hi["output"] / p_lo["output"]
            result.append(
                {
                    "from_model": hi,
                    "from_label": p_hi["label"],
                    "to_model": lo,
                    "to_label": p_lo["label"],
                    "input_mult": round(input_mult, 1),
                    "output_mult": round(output_mult, 1),
                }
            )
    return result


def parse_anthropic_usage(usage: dict[str, Any]) -> Usage:
    # Anthropic bundles extended-thinking tokens into output_tokens and does
    # not break them out in `usage`. We derive an estimate from the thinking
    # block text in derive_anthropic_thinking_tokens().
    return Usage(
        input_tokens=int(usage.get("input_tokens", 0) or 0),
        output_tokens=int(usage.get("output_tokens", 0) or 0),
        cache_read_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
        cache_creation_tokens=int(usage.get("cache_creation_input_tokens", 0) or 0),
    )


def parse_openai_usage(usage: dict[str, Any]) -> Usage:
    cached = 0
    thinking = 0
    details = usage.get("prompt_tokens_details") or {}
    if isinstance(details, dict):
        cached = int(details.get("cached_tokens", 0) or 0)
    completion_details = usage.get("completion_tokens_details") or {}
    if isinstance(completion_details, dict):
        thinking = int(completion_details.get("reasoning_tokens", 0) or 0)
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    return Usage(
        input_tokens=max(prompt - cached, 0),
        output_tokens=int(usage.get("completion_tokens", 0) or 0),
        cache_read_tokens=cached,
        thinking_tokens=thinking,
    )


def derive_anthropic_thinking_tokens(response_payload: Optional[dict[str, Any]]) -> int:
    # Anthropic's `usage` doesn't break out thinking tokens (they're folded
    # into `output_tokens`). Estimate from the assembled thinking-block text
    # using ~4 chars/token — close enough for a "incl. ~N thinking" hint.
    if not isinstance(response_payload, dict):
        return 0
    content = response_payload.get("content")
    if not isinstance(content, list):
        return 0
    chars = sum(
        len(b.get("thinking") or "")
        for b in content
        if isinstance(b, dict) and b.get("type") == "thinking"
    )
    return chars // 4


def parse_full_response_usage(payload: dict[str, Any]) -> Usage:
    if isinstance(payload, dict):
        u = payload.get("usage")
        if isinstance(u, dict):
            if "input_tokens" in u or "cache_read_input_tokens" in u:
                return parse_anthropic_usage(u)
            if "prompt_tokens" in u or "completion_tokens" in u:
                return parse_openai_usage(u)
    return Usage()


# ---------- transcript capture -----------------------------------------------


def detect_format(path: str) -> Optional[str]:
    """`anthropic` for /v1/messages, `openai` for /v1/chat/completions."""
    if "chat/completions" in path:
        return "openai"
    if "messages" in path:
        return "anthropic"
    return None


def parse_request(body_bytes: bytes, fmt: str) -> Optional[dict[str, Any]]:
    """Extract the conversation shape from a request body."""
    try:
        body = json.loads(body_bytes)
    except Exception:
        return None
    if not isinstance(body, dict):
        return None
    out: dict[str, Any] = {
        "format": fmt,
        "model": body.get("model"),
        "messages": list(body.get("messages") or []),
        "tools": list(body.get("tools") or []),
        "system": None,
    }
    if fmt == "anthropic":
        out["system"] = body.get("system")
    elif fmt == "openai":
        # OpenAI carries the system prompt as the first message in the array.
        msgs = out["messages"]
        if msgs and (msgs[0] or {}).get("role") == "system":
            out["system"] = (msgs[0] or {}).get("content")
            out["messages"] = msgs[1:]
    return out


def parse_full_response_content(payload: dict[str, Any], fmt: str) -> Optional[dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    if fmt == "anthropic":
        return {
            "role": "assistant",
            "content": payload.get("content") or [],
            "stop_reason": payload.get("stop_reason"),
        }
    if fmt == "openai":
        choices = payload.get("choices") or []
        if not choices:
            return None
        msg = (choices[0] or {}).get("message") or {}
        return {
            "role": "assistant",
            "content": msg.get("content"),
            "reasoning_content": msg.get("reasoning_content"),
            "tool_calls": msg.get("tool_calls"),
            "finish_reason": (choices[0] or {}).get("finish_reason"),
        }
    return None


class StreamAccumulator:
    """Parse SSE events from upstream and reconstruct usage + response content."""

    def __init__(self, fmt: str) -> None:
        self.fmt = fmt
        self.usage = Usage()
        # Anthropic state
        self._anth_blocks: dict[int, dict[str, Any]] = {}
        self._anth_partial_json: dict[int, str] = {}
        self._anth_stop_reason: Optional[str] = None
        # OpenAI state
        self._oai_text = ""
        self._oai_reasoning_text = ""
        self._oai_tool_calls: dict[int, dict[str, Any]] = {}
        self._oai_finish_reason: Optional[str] = None

    def feed_line(self, line: str) -> None:
        if not line.startswith("data:"):
            return
        body = line[5:].strip()
        if not body or body == "[DONE]":
            return
        try:
            evt = json.loads(body)
        except json.JSONDecodeError:
            return
        if self.fmt == "anthropic":
            self._feed_anthropic(evt)
        elif self.fmt == "openai":
            self._feed_openai(evt)

    def _feed_anthropic(self, evt: dict[str, Any]) -> None:
        etype = evt.get("type")
        if etype == "message_start":
            msg = evt.get("message") or {}
            u = msg.get("usage") or {}
            if u:
                parsed = parse_anthropic_usage(u)
                self.usage.input_tokens = parsed.input_tokens
                self.usage.cache_read_tokens = parsed.cache_read_tokens
                self.usage.cache_creation_tokens = parsed.cache_creation_tokens
                # message_start often reports output_tokens=1 placeholder; ignore.
        elif etype == "message_delta":
            u = evt.get("usage") or {}
            if "output_tokens" in u:
                self.usage.output_tokens = int(u["output_tokens"] or 0)
            delta = evt.get("delta") or {}
            if "stop_reason" in delta:
                self._anth_stop_reason = delta["stop_reason"]
        elif etype == "content_block_start":
            idx = evt.get("index", 0)
            block = dict(evt.get("content_block") or {})
            if block.get("type") == "text":
                block.setdefault("text", "")
            elif block.get("type") == "tool_use":
                block.setdefault("input", {})
            elif block.get("type") == "thinking":
                block.setdefault("thinking", "")
            self._anth_blocks[idx] = block
            self._anth_partial_json[idx] = ""
        elif etype == "content_block_delta":
            idx = evt.get("index", 0)
            d = evt.get("delta") or {}
            block = self._anth_blocks.get(idx)
            if block is None:
                return
            if d.get("type") == "text_delta":
                block["text"] = (block.get("text") or "") + (d.get("text") or "")
            elif d.get("type") == "input_json_delta":
                self._anth_partial_json[idx] = (
                    self._anth_partial_json.get(idx, "") + (d.get("partial_json") or "")
                )
            elif d.get("type") == "thinking_delta":
                block["thinking"] = (block.get("thinking") or "") + (d.get("thinking") or "")
        elif etype == "content_block_stop":
            idx = evt.get("index", 0)
            block = self._anth_blocks.get(idx)
            if block and block.get("type") == "tool_use":
                pj = self._anth_partial_json.get(idx, "")
                if pj:
                    try:
                        block["input"] = json.loads(pj)
                    except json.JSONDecodeError:
                        block["input"] = {"_raw_partial_json": pj}

    def _feed_openai(self, evt: dict[str, Any]) -> None:
        if "usage" in evt and isinstance(evt["usage"], dict):
            parsed = parse_openai_usage(evt["usage"])
            if not parsed.is_empty():
                self.usage.input_tokens = parsed.input_tokens
                self.usage.output_tokens = parsed.output_tokens
                self.usage.cache_read_tokens = parsed.cache_read_tokens
                self.usage.thinking_tokens = parsed.thinking_tokens
        choices = evt.get("choices") or []
        if not choices:
            return
        ch = choices[0] or {}
        delta = ch.get("delta") or {}
        if delta.get("content"):
            self._oai_text += delta["content"]
        if delta.get("reasoning_content"):
            self._oai_reasoning_text += delta["reasoning_content"]
        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            existing = self._oai_tool_calls.setdefault(
                idx,
                {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
            )
            if tc.get("id"):
                existing["id"] = tc["id"]
            if tc.get("type"):
                existing["type"] = tc["type"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                existing["function"]["name"] += fn["name"]
            if fn.get("arguments"):
                existing["function"]["arguments"] += fn["arguments"]
        if ch.get("finish_reason"):
            self._oai_finish_reason = ch["finish_reason"]

    def to_response(self) -> Optional[dict[str, Any]]:
        if self.fmt == "anthropic":
            blocks = [self._anth_blocks[k] for k in sorted(self._anth_blocks)]
            return {
                "role": "assistant",
                "content": blocks,
                "stop_reason": self._anth_stop_reason,
            }
        if self.fmt == "openai":
            tc = [self._oai_tool_calls[k] for k in sorted(self._oai_tool_calls)] or None
            return {
                "role": "assistant",
                "content": self._oai_text or None,
                "reasoning_content": self._oai_reasoning_text or None,
                "tool_calls": tc,
                "finish_reason": self._oai_finish_reason,
            }
        return None


# ---------- replay diff ------------------------------------------------------


def _msg_signature(msg: Any) -> str:
    try:
        return json.dumps(msg, sort_keys=True)
    except Exception:
        return repr(msg)


def compute_replay(curr_msgs: list, prev_msgs: Optional[list]) -> dict[str, Any]:
    """Find the longest common prefix; everything after is new this turn.

    Coding agents append new turns to the end of `messages`, so a prefix
    diff captures the "X messages replayed, Y new" split exactly.
    """
    if not prev_msgs:
        return {
            "new_count": len(curr_msgs),
            "replayed_count": 0,
            "new_indices": list(range(len(curr_msgs))),
            "first_request": True,
        }
    common = 0
    for a, b in zip(curr_msgs, prev_msgs):
        if _msg_signature(a) == _msg_signature(b):
            common += 1
        else:
            break
    return {
        "new_count": len(curr_msgs) - common,
        "replayed_count": common,
        "new_indices": list(range(common, len(curr_msgs))),
        "first_request": False,
    }


# ---------- stats ------------------------------------------------------------


@dataclass
class Stats:
    total: Usage = field(default_factory=Usage)
    requests: deque = field(default_factory=lambda: deque(maxlen=200))
    transcripts: dict[int, dict[str, Any]] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    request_seq: int = 0
    last_messages: Optional[list] = None
    subscribers: list[asyncio.Queue] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "totals": {
                **self.total.to_dict(),
                "costs": _costs_for(self.total),
            },
            "pricing": PRICING,
            "multipliers": compute_multipliers(),
            "started_at": self.started_at,
            "recent": list(self.requests)[:50],
        }

    def record(
        self,
        *,
        path: str,
        model: Optional[str],
        usage: Usage,
        duration_ms: int,
        request_payload: Optional[dict[str, Any]],
        response_payload: Optional[dict[str, Any]],
    ) -> Optional[int]:
        # Skip empty proxy hits (health checks etc.) but keep real LLM
        # requests even if usage parsing failed (we still want the transcript).
        if usage.is_empty() and not request_payload:
            return None

        if usage.thinking_tokens == 0 and (request_payload or {}).get("format") == "anthropic":
            usage.thinking_tokens = derive_anthropic_thinking_tokens(response_payload)

        self.total.merge(usage)
        self.request_seq += 1
        rid = self.request_seq

        msgs = (request_payload or {}).get("messages") or []
        replay = compute_replay(msgs, self.last_messages)
        self.last_messages = msgs

        rec = {
            "id": rid,
            "ts": time.time(),
            "path": path,
            "model": model or (request_payload or {}).get("model"),
            "format": (request_payload or {}).get("format"),
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_tokens": usage.cache_read_tokens,
            "cache_creation_tokens": usage.cache_creation_tokens,
            "thinking_tokens": usage.thinking_tokens,
            "costs": _costs_for(usage),
            "duration_ms": duration_ms,
            "message_count": len(msgs),
            "replay": {
                "new_count": replay["new_count"],
                "replayed_count": replay["replayed_count"],
                "first_request": replay["first_request"],
            },
        }
        self.requests.appendleft(rec)
        self.transcripts[rid] = {
            "id": rid,
            "summary": rec,
            "request": request_payload,
            "response": response_payload,
            "replay": replay,
        }
        # Trim stored transcripts to match the deque
        if self.requests.maxlen is not None and len(self.transcripts) > self.requests.maxlen:
            keep = {r["id"] for r in self.requests}
            for k in list(self.transcripts):
                if k not in keep:
                    del self.transcripts[k]
        self._broadcast()
        return rid

    def get_transcript(self, rid: int) -> Optional[dict[str, Any]]:
        return self.transcripts.get(rid)

    def reset(self) -> None:
        self.total = Usage()
        self.requests.clear()
        self.transcripts.clear()
        self.started_at = time.time()
        self.request_seq = 0
        self.last_messages = None
        self._broadcast()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        self.subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self.subscribers.remove(q)
        except ValueError:
            pass

    def _broadcast(self) -> None:
        snap = self.snapshot()
        for q in list(self.subscribers):
            try:
                q.put_nowait(snap)
            except asyncio.QueueFull:
                pass


# ---------- app --------------------------------------------------------------


_HOP_BY_HOP = {
    "host",
    "content-length",
    "connection",
    "transfer-encoding",
    "keep-alive",
    "upgrade",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
}


def _filter_request_headers(headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


def _filter_response_headers(headers) -> list[tuple[str, str]]:
    return [(k, v) for k, v in headers.items() if k.lower() not in _HOP_BY_HOP]


def build_app(*, upstream: str, backend: str = "openai_anthropic") -> FastAPI:
    app = FastAPI()
    stats = Stats()
    upstream = upstream.rstrip("/")
    client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=600, write=60, pool=10))

    @app.on_event("shutdown")
    async def _close() -> None:
        await client.aclose()

    # ----- dashboard / API ---------------------------------------------

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/snapshot")
    async def snapshot() -> JSONResponse:
        return JSONResponse(stats.snapshot())

    @app.post("/api/reset")
    async def reset() -> JSONResponse:
        stats.reset()
        return JSONResponse({"ok": True})

    @app.get("/api/request/{rid}")
    async def request_detail(rid: int) -> JSONResponse:
        t = stats.get_transcript(rid)
        if t is None:
            raise HTTPException(404, f"request {rid} not found")
        return JSONResponse(t)

    @app.get("/api/events")
    async def events(request: Request) -> StreamingResponse:
        q = stats.subscribe()

        async def gen() -> AsyncIterator[bytes]:
            yield f"data: {json.dumps(stats.snapshot())}\n\n".encode()
            try:
                while True:
                    if await request.is_disconnected():
                        return
                    try:
                        snap = await asyncio.wait_for(q.get(), timeout=15.0)
                        yield f"data: {json.dumps(snap)}\n\n".encode()
                    except asyncio.TimeoutError:
                        yield b": ping\n\n"
            finally:
                stats.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream")

    # ----- proxy --------------------------------------------------------

    @app.api_route(
        "/{full_path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def proxy(full_path: str, request: Request) -> Response:
        if full_path.startswith("api/") or full_path == "":
            return Response(status_code=404)

        url = f"{upstream}/{full_path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"

        body = await request.body()
        headers = _filter_request_headers(request.headers)
        fmt = detect_format(full_path)
        request_payload = parse_request(body, fmt) if body and fmt else None
        model = (request_payload or {}).get("model")
        started = time.perf_counter()

        upstream_req = client.build_request(
            request.method,
            url,
            headers=headers,
            content=body,
        )
        upstream_resp = await client.send(upstream_req, stream=True)

        ctype = upstream_resp.headers.get("content-type", "")
        is_sse = "text/event-stream" in ctype

        if is_sse and fmt:
            acc = StreamAccumulator(fmt)
            buf = ""

            async def relay_stream() -> AsyncIterator[bytes]:
                nonlocal buf
                try:
                    async for chunk in upstream_resp.aiter_raw():
                        if chunk:
                            try:
                                buf += chunk.decode("utf-8", errors="ignore")
                                while "\n" in buf:
                                    line, buf = buf.split("\n", 1)
                                    acc.feed_line(line.rstrip("\r"))
                            except Exception:
                                pass
                            yield chunk
                finally:
                    await upstream_resp.aclose()
                    duration_ms = int((time.perf_counter() - started) * 1000)
                    stats.record(
                        path="/" + full_path,
                        model=model,
                        usage=acc.usage,
                        duration_ms=duration_ms,
                        request_payload=request_payload,
                        response_payload=acc.to_response(),
                    )

            return StreamingResponse(
                relay_stream(),
                status_code=upstream_resp.status_code,
                headers=dict(_filter_response_headers(upstream_resp.headers)),
                media_type=ctype or "text/event-stream",
            )

        # Non-streaming
        raw = await upstream_resp.aread()
        await upstream_resp.aclose()
        duration_ms = int((time.perf_counter() - started) * 1000)
        usage = Usage()
        response_payload: Optional[dict[str, Any]] = None
        if "application/json" in ctype:
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    usage = parse_full_response_usage(payload)
                    if fmt:
                        response_payload = parse_full_response_content(payload, fmt)
            except json.JSONDecodeError:
                pass
        stats.record(
            path="/" + full_path,
            model=model,
            usage=usage,
            duration_ms=duration_ms,
            request_payload=request_payload,
            response_payload=response_payload,
        )

        return Response(
            content=raw,
            status_code=upstream_resp.status_code,
            headers=dict(_filter_response_headers(upstream_resp.headers)),
            media_type=ctype,
        )

    return app
