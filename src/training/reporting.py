"""Training-report file lifecycle helpers."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path


def archive_previous_report_files(
    log_file: Path,
    report_json: Path,
    *,
    resume: bool,
) -> list[tuple[Path, Path]]:
    """Archive report inputs for a new stage; resumed runs keep their history."""

    if resume or int(os.getenv("RANK", "0")) != 0:
        return []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archived: list[tuple[Path, Path]] = []
    for path in (log_file, report_json):
        if not path.is_file():
            continue
        destination = path.with_name(
            f"{path.stem}.previous-{timestamp}-{os.getpid()}{path.suffix}"
        )
        path.replace(destination)
        archived.append((path, destination))
    return archived
