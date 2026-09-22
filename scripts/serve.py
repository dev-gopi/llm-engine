"""Run the Gopi FastAPI service with Uvicorn."""

from __future__ import annotations

import sys

# Configure repository and ``src`` imports before Uvicorn imports the app.
# This also ensures the project-local ``datasets`` package is selected over
# the optional Hugging Face package with the same top-level name.
import _bootstrap  # noqa: F401

import argparse
import os

import uvicorn
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--config",
        nargs="?",
        const="configs/inference.yaml",
        default=None,
        help=(
            "Inference serving configuration file. If supplied without a value, "
            "uses configs/inference.yaml."
        ),
    )
    parser.add_argument(
        "--forwarded-allow-ips",
        default=os.getenv("GOPI_FORWARDED_ALLOW_IPS", "127.0.0.1"),
        help="Comma-separated trusted reverse-proxy IPs; use '*' only on an isolated network",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.reload and args.workers != 1:
        parser.error("--reload requires --workers 1")
    if args.config is not None:
        os.environ["GOPI_INFERENCE_CONFIG"] = args.config
    uvicorn.run(
        "serving.api:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level,
        proxy_headers=True,
        forwarded_allow_ips=args.forwarded_allow_ips,
    )


if __name__ == "__main__":
    main()
