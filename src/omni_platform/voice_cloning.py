"""Provider contract for voice cloning without coupling the core to a vendor."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .providers import GenerationResult, ProviderContext


class VoiceCloningProvider(Protocol):
    id: str
    capabilities: frozenset[str]
    def is_available(self) -> bool: ...
    def clone_voice(self, request: dict[str, Any], context: ProviderContext) -> GenerationResult: ...


@dataclass(frozen=True)
class VoiceClonePolicy:
    require_consent_token: bool = True
    max_reference_seconds: float = 30.0

    def validate(self, request: dict[str, Any]) -> None:
        if self.require_consent_token and not request.get("consent_token"):
            raise PermissionError("voice cloning requires an explicit consent token")
        duration = float(request.get("reference_duration_seconds", 0.0))
        if duration <= 0 or duration > self.max_reference_seconds:
            raise ValueError("reference_duration_seconds exceeds the configured safety limit")


class ProviderVoiceCloner:
    """Adapter wrapper that enforces policy before calling a vendor provider."""
    def __init__(self, provider: VoiceCloningProvider, policy: VoiceClonePolicy | None = None) -> None:
        self.provider = provider
        self.policy = policy or VoiceClonePolicy()

    def is_available(self) -> bool:
        return self.provider.is_available()

    def clone_voice(self, request: dict[str, Any], context: ProviderContext) -> GenerationResult:
        self.policy.validate(request)
        if not self.provider.is_available():
            raise RuntimeError(f"voice provider {self.provider.id!r} is unavailable")
        return self.provider.clone_voice(request, context)


__all__ = ["VoiceCloningProvider", "VoiceClonePolicy", "ProviderVoiceCloner"]
