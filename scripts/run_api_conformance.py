"""Run the release P0 API conformance suites.

Usage: python scripts/run_api_conformance.py
The script intentionally runs the repository's deterministic contract suites,
not a network-dependent external service.
"""
from __future__ import annotations

import subprocess
import sys

SUITES = [
    "tests/conformance/test_api.py",
    "tests/conformance/test_structured_outputs.py",
    "tests/conformance/test_tool_calling.py",
    "tests/tools/test_lifecycle.py",
    "tests/api/test_cancellation.py",
    "tests/mcp/test_contract.py",
    "tests/reasoning/test_budget.py",
    "tests/rag/test_reranker_contract.py",
    "tests/evaluation/test_rag_benchmark.py",
]


def main() -> int:
    command = [sys.executable, "-m", "pytest", "-q", *SUITES]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
