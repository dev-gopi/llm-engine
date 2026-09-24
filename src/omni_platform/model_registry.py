"""Unified multimodal model registry with runtime availability states."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

ModelStatus = Literal["configured", "installed", "loading", "ready", "unavailable", "error"]


@dataclass(slots=True)
class OmniModelRecord:
    id: str
    family: str
    provider: str
    task: str
    input_modalities: tuple[str, ...]
    output_modalities: tuple[str, ...]
    capabilities: dict[str, bool]
    status: ModelStatus = "configured"
    device: str | None = None
    precision: str | None = None
    context_length: int | None = None
    loaded: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["modalities"] = {
            "input": list(value.pop("input_modalities")),
            "output": list(value.pop("output_modalities")),
        }
        return value


class OmniModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, OmniModelRecord] = {}

    def register(self, record: OmniModelRecord) -> None:
        if not record.id or record.id in self._models:
            raise ValueError(f"duplicate or empty model id: {record.id!r}")
        self._models[record.id] = record

    def get(self, model_id: str) -> OmniModelRecord:
        return self._models[model_id]

    def update_status(self, model_id: str, *, status: ModelStatus, loaded: bool | None = None, device: str | None = None, precision: str | None = None) -> OmniModelRecord:
        record = self.get(model_id)
        record.status = status
        if loaded is not None:
            record.loaded = loaded
        if device is not None:
            record.device = device
        if precision is not None:
            record.precision = precision
        return record

    def list(self) -> list[OmniModelRecord]:
        return list(self._models.values())

    def capabilities(self) -> frozenset[str]:
        result: set[str] = set()
        for record in self._models.values():
            if record.status == "ready":
                result.update(name for name, enabled in record.capabilities.items() if enabled)
        return frozenset(result)
