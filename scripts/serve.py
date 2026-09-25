"""Run the Gopi FastAPI service with Uvicorn."""

from __future__ import annotations

import sys
# from pathlib import Path

# script_directory = str(Path(__file__).resolve().parent)
# if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
#     sys.path.pop(0)
# Configure repository and ``src`` imports before Uvicorn imports the app.
# This also ensures the project-local ``datasets`` package is selected over
# the optional Hugging Face package with the same top-level name.
import _bootstrap  # noqa: F401

import argparse
import os

import uvicorn
from dotenv import load_dotenv

from utils.logger import configure_logging, get_logger

logger = get_logger(__name__)


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
    configure_logging(level=args.log_level)
    logger.info("Starting serving backend on http://%s:%d (workers=%d)", args.host, args.port, args.workers)
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
