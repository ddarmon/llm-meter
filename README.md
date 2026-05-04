# llm-meter

A lightweight token / cost meter and live dashboard for LLM coding
agents. Two ways to use it:

1.  **Reverse-proxy** OpenAI / Anthropic-compatible traffic (e.g. a
    local oMLX server, the Anthropic API) and meter every request as it
    streams.
2.  **Read Claude Code session logs** directly from `~/.claude/projects`
    --- one-shot ingest, single-session view, or live tailing --- to see
    exactly what your usage *would* cost on the metered API. Useful when
    you're on a Pro/Max subscription and want to know how much you'd be
    paying per call.

The dashboard shows total cost, per-request breakdown (uncached input,
output, cache reads, cache writes split by 5m / 1h TTL), full
transcripts with replay analysis, and an optional flat-rate-vs-API
differential card.

## Quick start

### Mode A --- meter a local model server

```bash
# Start oMLX (your local model server) on port 8000
omlx-cli serve --port 8000
```

```bash
# Start the proxy/dashboard
uv run python -m llm_meter --port 8001 --upstream http://127.0.0.1:8000
```

```bash
# Launch your coding agent through the proxy
omlx-cli launch pi \
  --port 8001 \
  --model mlx-community/gemma-4-26b-a4b-it-4bit \
  --api-key "<YOUR_API_KEY>"
```

### Mode B --- meter Claude Code without a proxy

```bash
# Replay every Claude Code session you've ever run, then keep watching for
# new turns as you use it. Compare against a $200/mo Max plan.
uv run python -m llm_meter --port 8001 \
  --watch \
  --flat-rate 200 \
  --flat-rate-label "Claude Max \$200"
```

```bash
# Or scope to one specific session UUID (see `sesh sessions` if installed)
uv run python -m llm_meter --port 8001 \
  --session b51ab4c8-5d92-436e-ab47-2ff98bffc6d7 \
  --flat-rate 200
```

Open **http://localhost:8001/** in your browser to see the live
dashboard. Modes A and B can be combined: pass both `--upstream` and
`--watch` and the dashboard shows the union.

## Installation

```bash
git clone https://github.com/ddarmon/llm-meter.git
cd llm-meter
uv sync
```

Requires Python ≥ 3.11. No other manual dependencies --- managed by
`uv`.

## How it works

There are two data paths into the same `Stats` store, both rendered by
the same dashboard.

### Proxy path

```
Client → llm-meter (port 8001) → Upstream (oMLX / Anthropic, port 8000)
           ↑ captures & meters
           └──→ Live dashboard (real-time SSE updates)
```

1.  **Reverse-proxies** all API calls to your upstream model server,
    passing request and response bodies through unchanged.
2.  **Accumulates** streaming SSE events to reconstruct the full
    response (text + tool calls + usage metadata, including the 5m / 1h
    cache-write split when the upstream emits it).
3.  **Computes replay diffs** between consecutive turns in a session ---
    how many messages were replayed from history vs. new this turn.
4.  **Calculates costs** using built-in pricing tables (input / output /
    cache-read / cache-write 5m / cache-write 1h).
5.  **Streams** live snapshots to a single-page dashboard via
    Server-Sent Events.

### Session-log path

```
~/.claude/projects/<encoded-cwd>/<session-id>.jsonl
       │
       └──→ llm-meter (--ingest / --session / --watch) → same Stats / dashboard
```

Claude Code writes one JSONL file per session. `llm-meter` parses these
directly (no proxy needed):

-   **Dedupes** assistant records by `requestId` so the same API call
    isn't counted multiple times when its response is split across
    content blocks.
-   **Reconstructs** the request payload sent on the wire (the system
    prompt isn't stored, but its tokens are reflected in `usage`).
-   **Preserves** the cache-creation 5m / 1h split that Claude Code
    emits, so cost math matches what Anthropic would have billed.
-   **Tails** new turns as they're appended (`--watch`), so the
    dashboard updates live as you use Claude Code.

### What you get

| Feature                          | Description                                                                                                                                                                           |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Live cost card**               | Real-time dollar amount, updated per streaming response or per JSONL turn                                                                                                             |
| **Token meters**                 | Cache-read, cache-write, input (uncached), and output token counts                                                                                                                    |
| **Replay detection**             | Shows how many messages were replayed vs. new each turn (session-aware: parallel sessions don't contaminate each other's diffs)                                                       |
| **Cache monitoring**             | Tracks reads, 5m writes, and 1h writes separately and prices each correctly (1h = 2× input vs 1.25× for 5m). On the OpenAI Chat Completions path the cache-write slot displays "n/a". |
| **Full transcript viewer**       | Click any request to inspect messages, tool calls, system prompt                                                                                                                      |
| **Per-turn / cumulative toggle** | Switch between aggregate view and per-request cost                                                                                                                                    |
| **Model comparison**             | Select different Claude models to see what costs would be                                                                                                                             |
| **Flat-rate differential**       | Optional card comparing API equivalent vs your subscription cost (works for any flat rate — Pro, Max, Teams, etc.)                                                                    |
| **Multi-source**                 | Proxy traffic, JSONL ingest, and live tailing all feed the same dashboard                                                                                                             |

## Dashboard

The dashboard at `http://<host>:<port>/` shows:

-   **Header** --- live status dot, cumulative / per-turn toggle, reset
    button
-   **Cost card** --- total dollar amount (or per-turn), selected
    model's rates
-   **Meters** --- four tiles in prompt-cost order (cache reads, cache
    writes, input uncached, output) with sub-metrics
-   **Request feed** --- most recent requests with model, tokens, cost;
    click any row for the full transcript modal

The transcript modal shows:

-   Summary grid: cache read, cache write, input (uncached), output,
    cost (cache-write cell shows "n/a" for OpenAI-protocol requests)
-   Replay banner (new vs. replayed message count)
-   System prompt (expandable, with cache indicator)
-   All messages sent as input this turn (thick border = new, thin =
    replayed)
-   Assistant response (amber border, always new this turn)
-   Tool call and tool result rendering

## CLI reference

```
uv run python -m llm_meter [OPTIONS]
```

### Proxy

| Option       | Default                 | Description                                                                  |
| ------------ | ----------------------- | ---------------------------------------------------------------------------- |
| `--host`     | `127.0.0.1`             | Host to bind the proxy/dashboard                                             |
| `--port`     | `8001`                  | Port to bind                                                                 |
| `--upstream` | `http://127.0.0.1:8000` | Upstream backend base URL (env: `LLM_METER_UPSTREAM`)                        |
| `--backend`  | `openai_anthropic`      | Backend protocol: `openai_anthropic` or `bedrock` (env: `LLM_METER_BACKEND`) |

### Session-log ingest

| Option                 | Default                                     | Description                                                                               |
| ---------------------- | ------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `--watch [DIR]`        | off (default `~/.claude/projects` when set) | Tail Claude Code session logs and push new turns to the dashboard live                    |
| `--ingest PATH`        | off                                         | One-shot import of a JSONL file or directory of session logs                              |
| `--session ID`         | off                                         | Resolve a single session UUID and ingest just that file (use `sesh sessions` to list IDs) |
| `--projects-dir DIR`   | `~/.claude/projects`                        | Root dir to resolve `--session` IDs against                                               |
| `--no-replay-existing` | off                                         | With `--watch`, skip historical records and only show new turns                           |

### Subscription comparison

| Option                   | Default          | Description                                                                                      |
| ------------------------ | ---------------- | ------------------------------------------------------------------------------------------------ |
| `--flat-rate USD`        | off              | Monthly flat-rate cost (e.g. 20 for Pro, 200 for Max). Adds a differential card to the dashboard |
| `--flat-rate-label NAME` | `Flat-rate plan` | Label shown on the subscription card                                                             |

## API reference

| Endpoint            | Method | Description                                                                 |
| ------------------- | ------ | --------------------------------------------------------------------------- |
| `/`                 | GET    | Dashboard page                                                              |
| `/api/snapshot`     | GET    | Current totals, pricing table, and recent requests                          |
| `/api/events`       | GET    | SSE stream — pushes live snapshots (15s keepalive ping)                     |
| `/api/reset`        | POST   | Reset all counters and stored transcripts                                   |
| `/api/request/{id}` | GET    | Full transcript (summary + request + response + replay diff) for request ID |

## Project structure

```
llm-meter/
├── pyproject.toml          # Project config (FastAPI, uvicorn, httpx)
├── llm_meter/
│   ├── __main__.py         # CLI entry point (argparse + uvicorn + tailer thread)
│   ├── __init__.py         # Version
│   ├── server.py           # FastAPI app, proxy, Stats, pricing, SSE
│   ├── session_logs.py     # Claude Code JSONL parser + live tailer
│   └── static/
│       └── index.html      # Single-page dashboard (no JS dependencies)
└── uv.lock                 # Dependency lockfile
```

## Future ideas

-   [ ] Configurable pricing (load from file or env vars)
-   [ ] AWS Bedrock support (stubs exist, full support
    `--backend bedrock`)
-   [ ] Group dashboard by session / day
-   [ ] Token count comparison across models
-   [ ] Export / import transcript history
-   [ ] Cost alerts / budget caps
