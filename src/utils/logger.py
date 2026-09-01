"""Central logging configuration for the LLM engine."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(
    level: str | int | None = None,
    *,
    log_file: str | Path | None = None,
) -> None:
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
    if log_file is not None:
        destination = Path(log_file).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        already_configured = any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename).resolve() == destination
            for handler in project_logger.handlers
        )
        if not already_configured:
            file_handler = logging.FileHandler(destination, mode="a", encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(_FORMAT))
            project_logger.addHandler(file_handler)
    project_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced project logger."""
    configure_logging()
    return logging.getLogger(f"llm_engine.{name.removeprefix('src.')}")


logger = get_logger("llm")
