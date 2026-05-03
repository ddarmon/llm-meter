"""Reads Claude Code session JSONL logs from ~/.claude/projects and yields
per-turn records that match the shape llm-meter's Stats expects.

Each Claude Code session writes one JSONL file per session id. Records are
appended in time order. Relevant record types:

  - user        {message: {role:"user", content: str|list[block]}, ...}
  - assistant   {message: {model, content:list[block], usage:{...}},
                 requestId, ...}

Other record types (permission-mode, file-history-snapshot, attachment,
system, ai-title, last-prompt) carry no usage and don't affect the
reconstructed request payload, so they're ignored.

Two non-obvious quirks of the format that this module handles:

  * Multiple assistant records can share the same `requestId`. Each one
    holds a subset of the content blocks for a single API response, but
    the `usage` object on each is the response total (identical across
    duplicates). We dedupe by requestId so usage is counted exactly once
    and content blocks are merged in arrival order.

  * `cache_creation` is split into `ephemeral_5m_input_tokens` and
    `ephemeral_1h_input_tokens`. The 1h TTL is priced 2x input vs 1.25x
    for 5m, and Claude Code in practice uses 1h almost exclusively, so
    the split must be preserved (collapsing both into one bucket
    underprices every turn).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

DEFAULT_PROJECTS_DIR = Path.home() / ".claude" / "projects"


@dataclass
class TurnUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_5m_tokens: int = 0
    cache_creation_1h_tokens: int = 0

    @property
    def cache_creation_tokens(self) -> int:
        return self.cache_creation_5m_tokens + self.cache_creation_1h_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_5m_tokens": self.cache_creation_5m_tokens,
            "cache_creation_1h_tokens": self.cache_creation_1h_tokens,
        }


@dataclass
class SessionTurn:
    session_id: str
    request_id: str
    timestamp: float
    model: str
    cwd: str
    git_branch: Optional[str]
    is_sidechain: bool
    # Request-side history *before* this assistant turn — what was sent
    # on the wire (modulo system prompt, which the JSONL omits).
    messages: list[dict]
    # Anthropic-shaped assistant message (content blocks, stop_reason).
    response: dict
    usage: TurnUsage
    source_file: Path


def _parse_ts(s: Optional[str]) -> float:
    if not s:
        return time.time()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return time.time()


def _extract_usage(usage_obj: dict) -> TurnUsage:
    cc = usage_obj.get("cache_creation") or {}
    return TurnUsage(
        input_tokens=int(usage_obj.get("input_tokens", 0) or 0),
        output_tokens=int(usage_obj.get("output_tokens", 0) or 0),
        cache_read_tokens=int(usage_obj.get("cache_read_input_tokens", 0) or 0),
        cache_creation_5m_tokens=int(cc.get("ephemeral_5m_input_tokens", 0) or 0),
        cache_creation_1h_tokens=int(cc.get("ephemeral_1h_input_tokens", 0) or 0),
    )


@dataclass
class _ParseState:
    """Cross-record state for one session file (or one tailed file)."""

    history: list[dict] = field(default_factory=list)
    seen_request_ids: set[str] = field(default_factory=set)
    # Live response dicts keyed by requestId so duplicate records can
    # extend content blocks of the already-yielded turn in place.
    response_by_request: dict[str, dict] = field(default_factory=dict)


def _consume(rec: dict, state: _ParseState, source: Path) -> Iterator[SessionTurn]:
    t = rec.get("type")
    if t == "user":
        msg = rec.get("message") or {}
        if msg.get("role") == "user" and msg.get("content") is not None:
            state.history.append({"role": "user", "content": msg["content"]})
        return
    if t != "assistant":
        return

    msg = rec.get("message") or {}
    req_id = rec.get("requestId")
    if not req_id:
        return
    content = msg.get("content") or []

    if req_id in state.seen_request_ids:
        resp = state.response_by_request.get(req_id)
        if resp is not None and isinstance(content, list):
            resp["content"].extend(content)
        return

    state.seen_request_ids.add(req_id)
    response = {
        "role": "assistant",
        "content": list(content) if isinstance(content, list) else [],
        "stop_reason": msg.get("stop_reason"),
    }
    state.response_by_request[req_id] = response

    turn = SessionTurn(
        session_id=rec.get("sessionId", ""),
        request_id=req_id,
        timestamp=_parse_ts(rec.get("timestamp")),
        model=msg.get("model", ""),
        cwd=rec.get("cwd", ""),
        git_branch=rec.get("gitBranch"),
        is_sidechain=bool(rec.get("isSidechain", False)),
        # Snapshot history *before* we append the assistant turn; that's
        # exactly the request payload that was sent for this API call.
        messages=list(state.history),
        response=response,
        usage=_extract_usage(msg.get("usage") or {}),
        source_file=source,
    )
    # Aliases the live `response["content"]` list, so any duplicate
    # records that arrive later and extend it are reflected in future
    # request payloads built from `state.history` too.
    state.history.append({"role": "assistant", "content": response["content"]})
    yield turn


def iter_session_turns(path: Path) -> Iterator[SessionTurn]:
    """Walk one session JSONL file and yield one SessionTurn per unique requestId."""
    state = _ParseState()
    with path.open() as f:
        for raw in f:
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            yield from _consume(rec, state, path)


def find_session_file(
    session_id: str, root: Path = DEFAULT_PROJECTS_DIR
) -> Path:
    """Resolve a Claude Code session UUID to its JSONL path under `root`.

    Raises FileNotFoundError if no match, RuntimeError if more than one
    project dir contains a file with that name.
    """
    if not root.exists():
        raise FileNotFoundError(f"projects root not found: {root}")
    matches = list(root.rglob(f"{session_id}.jsonl"))
    if not matches:
        raise FileNotFoundError(f"session {session_id} not found under {root}")
    if len(matches) > 1:
        joined = "\n  ".join(str(m) for m in matches)
        raise RuntimeError(f"session id matches multiple files:\n  {joined}")
    return matches[0]


def iter_dir(root: Path = DEFAULT_PROJECTS_DIR) -> Iterator[SessionTurn]:
    """Yield turns from every .jsonl under root (mtime-ordered, oldest first).

    Each session file gets its own parse state, so there's no cross-session
    bleed even though Claude Code interleaves projects under the same root.
    """
    if not root.exists():
        return
    files = sorted(root.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    for f in files:
        yield from iter_session_turns(f)


# ---------- live tail --------------------------------------------------------


@dataclass
class _TailState:
    offset: int = 0
    inode: int = 0
    parse: _ParseState = field(default_factory=_ParseState)


def tail_dir(
    root: Path = DEFAULT_PROJECTS_DIR,
    *,
    poll_seconds: float = 0.5,
    stop_event: Optional[Any] = None,  # threading.Event-like
    replay_existing: bool = True,
) -> Iterator[SessionTurn]:
    """Poll-based tailer for newly-appended turns.

    On startup, if `replay_existing` is True (default), every existing turn
    in every .jsonl under `root` is yielded once, oldest file first, so the
    dashboard reflects historical usage. Thereafter only new records are
    emitted as they appear.

    `stop_event` is checked between polls; pass a threading.Event-like
    object to stop the loop cleanly.
    """
    states: dict[Path, _TailState] = {}

    if replay_existing and root.exists():
        for path in sorted(root.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime):
            state = _TailState(inode=path.stat().st_ino)
            states[path] = state
            yield from _read_from(path, state)

    while True:
        if stop_event is not None and stop_event.is_set():
            return
        if root.exists():
            for path in root.rglob("*.jsonl"):
                try:
                    st = path.stat()
                except FileNotFoundError:
                    continue
                state = states.get(path)
                if state is None:
                    state = _TailState(inode=st.st_ino)
                    states[path] = state
                elif state.inode != st.st_ino or st.st_size < state.offset:
                    # File rotated or truncated — start fresh.
                    state = _TailState(inode=st.st_ino)
                    states[path] = state
                if st.st_size == state.offset:
                    continue
                yield from _read_from(path, state)
        time.sleep(poll_seconds)


def _read_from(path: Path, state: _TailState) -> Iterator[SessionTurn]:
    # Use readline() (not `for raw in f`) so that f.tell() works between
    # lines — Python 3's file iterator buffers ahead and disables tell().
    with path.open() as f:
        f.seek(state.offset)
        while True:
            raw = f.readline()
            if not raw:
                break
            if not raw.endswith("\n"):
                # Partial line — leave offset where it was, re-read next poll.
                break
            state.offset = f.tell()
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            yield from _consume(rec, state.parse, path)


# ---------- smoke test ------------------------------------------------------


if __name__ == "__main__":
    import sys

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PROJECTS_DIR
    n = 0
    in_tot = out_tot = cr_tot = cw5 = cw1 = 0
    by_model: dict[str, int] = {}
    for turn in iter_dir(root):
        n += 1
        u = turn.usage
        in_tot += u.input_tokens
        out_tot += u.output_tokens
        cr_tot += u.cache_read_tokens
        cw5 += u.cache_creation_5m_tokens
        cw1 += u.cache_creation_1h_tokens
        by_model[turn.model] = by_model.get(turn.model, 0) + 1
    print(f"turns:           {n:,}")
    print(f"input:           {in_tot:,}")
    print(f"output:          {out_tot:,}")
    print(f"cache_read:      {cr_tot:,}")
    print(f"cache_write_5m:  {cw5:,}")
    print(f"cache_write_1h:  {cw1:,}")
    print("by model:")
    for m, c in sorted(by_model.items(), key=lambda kv: -kv[1]):
        print(f"  {c:>6,}  {m}")
