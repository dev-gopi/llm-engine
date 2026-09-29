"""Provider contract for voice cloning without coupling the core to a vendor."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

from .providers import GenerationResult, ProviderContext


class VoiceCloningProvider(Protocol):
    id: str
    capabilities: frozenset[str]

    def is_available(self) -> bool: ...
    def clone_voice(
        self, request: dict[str, Any], context: ProviderContext
    ) -> GenerationResult: ...


@dataclass(frozen=True)
class VoiceClonePolicy:
    require_consent_token: bool = True
    max_reference_seconds: float = 30.0

    def validate(self, request: dict[str, Any]) -> None:
        duration = float(request.get("reference_duration_seconds", 0.0))
        if duration <= 0 or duration > self.max_reference_seconds:
            raise ValueError(
                "reference_duration_seconds exceeds the configured safety limit"
            )
        if self.require_consent_token and not request.get("consent_verified"):
            raise PermissionError("voice cloning requires a verified consent token")


class ProviderVoiceCloner:
    """Adapter wrapper that enforces policy before calling a vendor provider."""

    def __init__(
        self, provider: VoiceCloningProvider, policy: VoiceClonePolicy | None = None
    ) -> None:
        self.provider = provider
        self.policy = policy or VoiceClonePolicy()

    def is_available(self) -> bool:
        return self.provider.is_available()

    def clone_voice(
        self, request: dict[str, Any], context: ProviderContext
    ) -> GenerationResult:
        self.policy.validate(request)
        if not self.provider.is_available():
            raise RuntimeError(f"voice provider {self.provider.id!r} is unavailable")
        return self.provider.clone_voice(request, context)


def verify_consent_token(
    token: str, *, secret: str, asset_id: str, tenant_id: str
) -> None:
    """Verify a short-lived HMAC consent assertion bound to asset and tenant."""
    try:
        encoded, signature = token.rsplit(".", 1)
        expected = hmac.new(
            secret.encode(), encoded.encode(), hashlib.sha256
        ).hexdigest()
        claims = json.loads(
            base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        )
    except Exception as exc:
        raise PermissionError("invalid voice-clone consent token") from exc
    if not hmac.compare_digest(signature, expected):
        raise PermissionError("invalid voice-clone consent token")
    if float(claims.get("exp", 0)) < time.time():
        raise PermissionError("voice-clone consent token has expired")
    if claims.get("asset_id") != asset_id or claims.get("tenant_id") != tenant_id:
        raise PermissionError("voice-clone consent token is not valid for this asset")


__all__ = [
    "VoiceCloningProvider",
    "VoiceClonePolicy",
    "ProviderVoiceCloner",
    "verify_consent_token",
]
