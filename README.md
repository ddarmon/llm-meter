# llm-meter

A lightweight reverse-proxy that meters token usage, cost, and conversation transcripts in real time for OpenAI / Anthropic-compatible LLM endpoints.

Point your coding agent at `llm-meter`, and it forwards traffic to your model server while displaying a live dashboard — total cost, per-request breakdown, cache hit rates, and full message transcripts with replay analysis.

## Quick start

```bash
# Start oMLX (your local model server) on port 8000
omlx-cli serve --port 8000
```

```bash
# Start the proxy/dashboard (sits between agent and model server)
uv run python -m llm_meter \
  --port 8001 \
  --upstream http://127.0.0.1:8000
```

```bash
# Launch your coding agent through the proxy
omlx-cli launch pi \
  --port 8001 \
  --model mlx-community/gemma-4-26b-a4b-it-4bit \
  --api-key "<YOUR_API_KEY>"
```

```bash
# Or with a different model
omlx-cli launch pi \
  --port 8001 \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --api-key "<YOUR_API_KEY>"
```

Open **http://localhost:8001/** in your browser to see the live dashboard.

## Installation

```bash
git clone https://github.com/ddarmon/llm-meter.git
cd llm-meter
uv sync
```

Requires Python ≥ 3.11. No other manual dependencies — managed by `uv`.

## How it works

```
Client → llm-meter (your proxy, port 8001) → Upstream (oMLX, port 8000)
           ↑ captures & meters
           └──→ Live dashboard (real-time SSE updates)
```

`llm-meter` sits as a man-in-the-middle:

1. **Reverse-proxies** all API calls to your upstream model server, passing request bodies and response bodies through unchanged.
2. **Accumulates** streaming SSE events to reconstruct the full response (text + tool calls + usage metadata).
3. **Computes replay diffs** between consecutive turns — how many messages were replayed from conversation history vs. new this turn.
4. **Calculates costs** using built-in pricing tables (configurable model rates, input/output/cache-read/write).
5. **Streams** live snapshots to a single-page dashboard via Server-Sent Events.

### What you get

| Feature | Description |
|---|---|
| **Live cost card** | Real-time dollar amount, updated per streaming response |
| **Token meters** | Cache-read, cache-write, input (uncached), and output token counts |
| **Replay detection** | Shows how many messages were replayed vs. new each turn |
| **Cache monitoring** | Tracks cache reads vs. writes, shows cached prefix %. On the OpenAI Chat Completions path the cache-write slot displays "n/a" because the usage spec doesn't expose writes. |
| **Full transcript viewer** | Click any request to inspect messages, tool calls, system prompt |
| **Per-turn / cumulative toggle** | Switch between aggregate view and per-request cost |
| **Model comparison** | Select different Claude models to see what costs would be |
| **Multi-backend** | Works with OpenAI, Anthropic, and Bedrock protocol families |

## Dashboard

The dashboard at `http://<host>:<port>/` shows:

- **Header** — live status dot, cumulative / per-turn toggle, reset button
- **Cost card** — total dollar amount (or per-turn), selected model's rates
- **Meters** — four tiles in prompt-cost order (cache reads, cache writes, input uncached, output) with sub-metrics
- **Request feed** — most recent requests with model, tokens, cost; click any row for the full transcript modal

The transcript modal shows:
- Summary grid: cache read, cache write, input (uncached), output, cost (cache-write cell shows "n/a" for OpenAI-protocol requests)
- Replay banner (new vs. replayed message count)
- System prompt (expandable, with cache indicator)
- All messages sent as input this turn (thick border = new, thin = replayed)
- Assistant response (amber border, always new this turn)
- Tool call and tool result rendering

## CLI reference

```
uv run python -m llm_meter [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Host to bind the proxy/dashboard |
| `--port` | `8001` | Port to bind |
| `--upstream` | `http://127.0.0.1:8000` | Upstream backend base URL (env: `LLM_METER_UPSTREAM`) |
| `--backend` | `openai_anthropic` | Backend protocol: `openai_anthropic` or `bedrock` (env: `LLM_METER_BACKEND`) |

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Dashboard page |
| `/api/snapshot` | GET | Current totals, pricing table, and recent requests |
| `/api/events` | GET | SSE stream — pushes live snapshots (15s keepalive ping) |
| `/api/reset` | POST | Reset all counters and stored transcripts |
| `/api/request/{id}` | GET | Full transcript (summary + request + response + replay diff) for request ID |

## Project structure

```
llm-meter/
├── pyproject.toml          # Project config (FastAPI, uvicorn, httpx)
├── llm_meter/
│   ├── __main__.py         # CLI entry point (argparse + uvicorn)
│   ├── __init__.py         # Version
│   ├── server.py           # FastAPI app, proxy logic, stats, SSE
│   └── static/
│       └── index.html      # Single-page dashboard (no JS dependencies)
└── uv.lock                 # Dependency lockfile
```

## Future ideas

- [ ] Configurable pricing (load from file or env vars)
- [ ] AWS Bedrock support (stubs exist, full support `--backend bedrock`)
- [ ] Request deduplication / caching of costs
- [ ] Token count comparison across models
- [ ] Export / import transcript history
- [ ] Cost alerts / budget caps
