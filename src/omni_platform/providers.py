"""Vendor-neutral provider contracts for multimodal generation and understanding."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

ProgressCallback = Callable[[float, str], None]


@dataclass(slots=True)
class ProviderContext:
    request_id: str
    user_id: str | None = None
    organization_id: str | None = None
    cancellation_token: Any | None = None
    progress: ProgressCallback | None = None


@dataclass(slots=True)
class GenerationArtifact:
    path: Path
    mime_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationResult:
    artifacts: list[GenerationArtifact]
    usage: dict[str, float | int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class BaseProvider(Protocol):
    id: str
    capabilities: frozenset[str]

    def is_available(self) -> bool: ...


@runtime_checkable
class ImageGenerationProvider(BaseProvider, Protocol):
    def generate_image(self, request: dict[str, Any], context: ProviderContext) -> GenerationResult: ...
    def edit_image(self, request: dict[str, Any], context: ProviderContext) -> GenerationResult: ...


@runtime_checkable
class AudioGenerationProvider(BaseProvider, Protocol):
    def generate_audio(self, request: dict[str, Any], context: ProviderContext) -> GenerationResult: ...


@runtime_checkable
class TextToSpeechProvider(BaseProvider, Protocol):
    def synthesize(self, request: dict[str, Any], context: ProviderContext) -> GenerationResult: ...


@runtime_checkable
class SpeechToTextProvider(BaseProvider, Protocol):
    def transcribe(self, request: dict[str, Any], context: ProviderContext) -> dict[str, Any]: ...


@runtime_checkable
class SpeechToSpeechProvider(BaseProvider, Protocol):
    def convert_speech(self, request: dict[str, Any], context: ProviderContext) -> GenerationResult: ...


@runtime_checkable
class AudioUnderstandingProvider(BaseProvider, Protocol):
    def understand_audio(self, request: dict[str, Any], context: ProviderContext) -> dict[str, Any]: ...


@runtime_checkable
class VideoGenerationProvider(BaseProvider, Protocol):
    def generate_video(self, request: dict[str, Any], context: ProviderContext) -> GenerationResult: ...
    def edit_video(self, request: dict[str, Any], context: ProviderContext) -> GenerationResult: ...


@runtime_checkable
class VideoUnderstandingProvider(BaseProvider, Protocol):
    def understand_video(self, request: dict[str, Any], context: ProviderContext) -> dict[str, Any]: ...


class ProviderRegistry:
    """Thread-safe-enough process-local provider registry with explicit capability lookup."""

    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {}

    def register(self, provider: BaseProvider) -> None:
        if not provider.id or provider.id in self._providers:
            raise ValueError(f"provider id must be unique and non-empty: {provider.id!r}")
        self._providers[provider.id] = provider

    def get(self, provider_id: str) -> BaseProvider:
        return self._providers[provider_id]

    def available(self, capability: str) -> list[BaseProvider]:
        return [
            provider
            for provider in self._providers.values()
            if provider.is_available() and capability in provider.capabilities
        ]

    def capability_set(self) -> frozenset[str]:
        values: set[str] = set()
        for provider in self._providers.values():
            if provider.is_available():
                values.update(provider.capabilities)
        return frozenset(values)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "id": p.id,
                "available": p.is_available(),
                "capabilities": sorted(p.capabilities),
            }
            for p in self._providers.values()
        ]
