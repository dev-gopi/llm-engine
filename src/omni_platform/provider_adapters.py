"""Generic interchangeable HTTP multimodal provider adapter.

This adapter is intentionally transport-neutral and requires callers to supply
an authenticated endpoint. It gives image/audio/video/understanding providers a
common lifecycle without baking a vendor into the core engine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable
from urllib.request import Request, urlopen

from .providers import ProviderContext


@dataclass(frozen=True)
class RemoteProviderConfig:
    id: str
    endpoint: str
    capabilities: frozenset[str]
    timeout_seconds: float = 60.0
    api_key: str | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.endpoint:
            raise ValueError("provider id and endpoint are required")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


class RemoteJSONProvider:
    """Small standard-library provider for interchangeable vendor endpoints."""

    def __init__(
        self,
        config: RemoteProviderConfig,
        *,
        transport: Callable[..., Any] | None = None,
    ) -> None:
        self.id = config.id
        self.capabilities = config.capabilities
        self.config = config
        self._transport = transport or urlopen

    def is_available(self) -> bool:
        return bool(self.config.endpoint)

    def invoke(
        self, operation: str, request: dict[str, Any], context: ProviderContext
    ) -> dict[str, Any]:
        if operation not in self.capabilities:
            raise ValueError(f"provider {self.id!r} does not advertise {operation!r}")
        body = json.dumps(
            {
                "operation": operation,
                "request": request,
                "request_id": context.request_id,
            }
        ).encode()
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        req = Request(self.config.endpoint, data=body, headers=headers, method="POST")
        with self._transport(req, timeout=self.config.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("provider response must be a JSON object")
        return payload


@dataclass(frozen=True)
class LatentImageProfile:
    name: str
    checkpoint: str
    vae: str
    text_encoder: str
    scheduler: str
    dtype: str = "float16"
    production_qualified: bool = False

    def validate(self) -> None:
        for field in (
            self.name,
            self.checkpoint,
            self.vae,
            self.text_encoder,
            self.scheduler,
        ):
            if not field.strip():
                raise ValueError("latent-image profile contains an empty component")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("unsupported latent-image dtype")


__all__ = ["RemoteProviderConfig", "RemoteJSONProvider", "LatentImageProfile"]
