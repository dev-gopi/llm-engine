"""Central logging configuration for the LLM engine."""

from __future__ import annotations

import logging
import os
import sys

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str | int | None = None) -> None:
    """Configure the project logger once without changing root handlers."""
    project_logger = logging.getLogger("llm_engine")
    resolved = level if level is not None else os.getenv("LOG_LEVEL", "INFO")
    if isinstance(resolved, str):
        numeric_level = logging.getLevelName(resolved.upper())
        if not isinstance(numeric_level, int):
            raise ValueError(f"invalid logging level: {resolved}")
    else:
        numeric_level = resolved
    project_logger.setLevel(numeric_level)
    if not project_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT))
        project_logger.addHandler(handler)
    project_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced project logger."""
    configure_logging()
    return logging.getLogger(f"llm_engine.{name.removeprefix('src.')}")


logger = get_logger("llm")
