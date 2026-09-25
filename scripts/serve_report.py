"""Launch a standalone dashboard server for real-time monitoring of generation, serving, chat, and RAG."""

from __future__ import annotations

import sys
from pathlib import Path

_script_dir = str(Path(__file__).resolve().parent)
sys.path[:] = [e for e in sys.path if str(Path(e or ".").resolve()) != _script_dir]

import argparse
import uvicorn
from fastapi import FastAPI
from serving.report_router import router as report_router

app = FastAPI(
    title="Gopi Engine System & Inference Dashboard",
    description="Live status, capacity, generation latency, chat history, and RAG statistics.",
    version="0.1.0",
)

app.include_router(report_router)

# Also expose / directly for root access
@app.get("/")
def root_redirect():
    from serving.report_router import get_dashboard
    return get_dashboard()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve dashboard on (default: 7860)")
    parser.add_argument("--log-level", default="info", help="Uvicorn log level (default: info)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    print("=" * 60)
    print("  🚀 Gopi Engine — System & Operations Dashboard")
    print(f"  🌐 Open browser at: http://localhost:{args.port}/")
    print("=" * 60)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
