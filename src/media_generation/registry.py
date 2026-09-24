"""Named media-model registry for production serving."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.config import load_yaml


@dataclass(frozen=True)
class MediaModelSpec:
    id: str
    kind: str
    config: Path
    checkpoint: Path
    description: str = ""
    revision: str = "1"
    enabled: bool = True
    capabilities: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def load_registry(path: str | Path) -> dict[str, MediaModelSpec]:
    payload = load_yaml(Path(path))
    models = payload.get("models", []) if isinstance(payload, dict) else []
    result: dict[str, MediaModelSpec] = {}
    for item in models:
        if not isinstance(item, dict):
            raise ValueError("media registry entries must be objects")
        capabilities = item.get("capabilities", []) or []
        if not isinstance(capabilities, list) or any(not isinstance(value, str) for value in capabilities):
            raise ValueError("media model capabilities must be a string list")
        metadata = item.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            raise ValueError("media model metadata must be an object")
        spec = MediaModelSpec(
            id=str(item["id"]),
            kind=str(item["kind"]),
            config=Path(item["config"]),
            checkpoint=Path(item["checkpoint"]),
            description=str(item.get("description", "")),
            revision=str(item.get("revision", "1")),
            enabled=bool(item.get("enabled", True)),
            capabilities=tuple(capabilities),
            metadata=dict(metadata),
        )
        if spec.kind not in {"audio", "video"}:
            raise ValueError(f"unsupported media model kind: {spec.kind}")
        if not spec.id or len(spec.id) > 128:
            raise ValueError("media model id must contain 1 to 128 characters")
        if spec.id in result:
            raise ValueError(f"duplicate media model id: {spec.id}")
        if spec.enabled:
            result[spec.id] = spec
    return result
