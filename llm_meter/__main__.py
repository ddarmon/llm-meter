import argparse
import os
import threading
from pathlib import Path

import uvicorn

from .server import Stats, build_app
from .session_logs import (
    DEFAULT_PROJECTS_DIR,
    find_session_file,
    iter_dir,
    iter_session_turns,
    tail_dir,
)


def _ingest_path(stats: Stats, target: Path) -> int:
    """One-shot ingest of a JSONL file or a directory of session files."""
    if target.is_file():
        turns = iter_session_turns(target)
    else:
        turns = iter_dir(target)
    n = 0
    for turn in turns:
        if stats.record_turn(turn) is not None:
            n += 1
    return n


def _start_tailer(stats: Stats, root: Path, replay_existing: bool) -> threading.Event:
    """Spawn a background daemon thread that tails session logs and feeds
    new turns into Stats. Returns the stop event for shutdown wiring."""
    stop = threading.Event()

    def run() -> None:
        for turn in tail_dir(root, stop_event=stop, replay_existing=replay_existing):
            try:
                stats.record_turn(turn)
            except Exception:
                # The tailer is best-effort; never let a parse error kill it.
                continue

    t = threading.Thread(target=run, name="llm-meter-tailer", daemon=True)
    t.start()
    return stop


def main() -> None:
    p = argparse.ArgumentParser(prog="llm-meter")
    p.add_argument("--host", default="127.0.0.1", help="Host to bind the proxy/dashboard")
    p.add_argument("--port", type=int, default=8001, help="Port to bind")
    p.add_argument(
        "--upstream",
        default=os.environ.get("LLM_METER_UPSTREAM", "http://127.0.0.1:8000"),
        help="Upstream backend base URL (e.g. http://127.0.0.1:8000 for oMLX)",
    )
    p.add_argument(
        "--backend",
        default=os.environ.get("LLM_METER_BACKEND", "openai_anthropic"),
        choices=["openai_anthropic", "bedrock"],
        help="Backend protocol family (controls how usage is parsed)",
    )
    p.add_argument(
        "--watch",
        nargs="?",
        const=str(DEFAULT_PROJECTS_DIR),
        default=None,
        metavar="DIR",
        help=(
            "Tail Claude Code session logs and feed turns into the dashboard "
            f"(default dir: {DEFAULT_PROJECTS_DIR})."
        ),
    )
    p.add_argument(
        "--ingest",
        default=None,
        metavar="PATH",
        help="One-shot ingest a JSONL file or directory of session logs at startup, then serve.",
    )
    p.add_argument(
        "--session",
        default=None,
        metavar="ID",
        help=(
            "Ingest one specific Claude Code session by its UUID. Resolved "
            "against the --projects-dir root (default ~/.claude/projects). "
            "Use `sesh sessions` to list available IDs."
        ),
    )
    p.add_argument(
        "--projects-dir",
        default=str(DEFAULT_PROJECTS_DIR),
        metavar="DIR",
        help="Root dir to resolve --session IDs against (default: ~/.claude/projects).",
    )
    p.add_argument(
        "--no-replay-existing",
        action="store_true",
        help="With --watch, skip historical records and only show new turns.",
    )
    p.add_argument(
        "--flat-rate",
        type=float,
        default=None,
        metavar="USD",
        help=(
            "Monthly flat-rate subscription cost in USD (e.g. 20 for Pro, "
            "100 or 200 for Max). Adds a subscription-differential card "
            "comparing API equivalent vs the flat rate."
        ),
    )
    p.add_argument(
        "--flat-rate-label",
        default="Flat-rate plan",
        metavar="NAME",
        help='Label for the subscription card (e.g. "Claude Max $200"). Default: "Flat-rate plan".',
    )
    args = p.parse_args()

    app, stats = build_app(
        upstream=args.upstream,
        backend=args.backend,
        flat_rate_usd=args.flat_rate,
        flat_rate_label=args.flat_rate_label,
    )

    if args.session:
        try:
            session_path = find_session_file(
                args.session, Path(args.projects_dir).expanduser()
            )
        except (FileNotFoundError, RuntimeError) as e:
            raise SystemExit(f"--session: {e}")
        n = _ingest_path(stats, session_path)
        print(f"llm-meter ingested {n} turns from session {args.session}")
        print(f"  source: {session_path}")

    if args.ingest:
        target = Path(args.ingest).expanduser()
        if not target.exists():
            raise SystemExit(f"--ingest path not found: {target}")
        n = _ingest_path(stats, target)
        print(f"llm-meter ingested {n} turns from {target}")

    if args.watch:
        watch_root = Path(args.watch).expanduser()
        _start_tailer(stats, watch_root, replay_existing=not args.no_replay_existing)
        print(f"llm-meter watching:   {watch_root}")

    print(f"llm-meter dashboard:  http://{args.host}:{args.port}/")
    print(f"llm-meter proxy:      http://{args.host}:{args.port}/  ->  {args.upstream}")
    if args.flat_rate is not None:
        print(
            f"llm-meter flat-rate:  ${args.flat_rate:.2f}/mo "
            f"({args.flat_rate_label}) — subscription comparison enabled"
        )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
