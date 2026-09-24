"""Unified generation job schema and legal state transitions."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

JobStatus = Literal["queued", "initializing", "processing", "postprocessing", "completed", "failed", "cancelled"]

_ALLOWED = {
    "queued": {"initializing", "cancelled", "failed"},
    "initializing": {"processing", "cancelled", "failed"},
    "processing": {"postprocessing", "cancelled", "failed"},
    "postprocessing": {"completed", "cancelled", "failed"},
    "completed": set(),
    "failed": set(),
    "cancelled": set(),
}


@dataclass(slots=True)
class GenerationJob:
    modality: str
    model: str
    provider: str
    prompt: str | None = None
    id: str = field(default_factory=lambda: f"gen_{uuid.uuid4().hex}")
    request_id: str | None = None
    organization_id: str | None = None
    user_id: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    progress: float = 0.0
    status: JobStatus = "queued"
    stage: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    error_code: str | None = None
    error_message: str | None = None
    retry_count: int = 0
    artifacts: list[str] = field(default_factory=list)
    resource_usage: dict[str, float | int] = field(default_factory=dict)

    def transition(self, status: JobStatus, *, progress: float | None = None, stage: str | None = None) -> None:
        if status not in _ALLOWED[self.status]:
            raise ValueError(f"illegal job transition {self.status!r} -> {status!r}")
        now = time.time()
        if self.status == "queued" and status == "initializing":
            self.started_at = now
        self.status = status
        self.stage = stage or status
        if progress is not None:
            self.progress = max(0.0, min(1.0, float(progress)))
        if status == "completed":
            self.progress = 1.0
            self.completed_at = now
        elif status in {"failed", "cancelled"}:
            self.completed_at = now

    @property
    def generation_duration(self) -> float | None:
        if self.started_at is None or self.completed_at is None:
            return None
        return max(0.0, self.completed_at - self.started_at)
