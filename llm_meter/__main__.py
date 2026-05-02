import argparse
import os

import uvicorn

from .server import build_app


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
    args = p.parse_args()

    app = build_app(upstream=args.upstream, backend=args.backend)
    print(f"llm-meter dashboard:  http://{args.host}:{args.port}/")
    print(f"llm-meter proxy:      http://{args.host}:{args.port}/  ->  {args.upstream}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
