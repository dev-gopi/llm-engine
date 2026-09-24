"""Runtime-derived capability snapshots for the unified multimodal API."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


KNOWN_CAPABILITIES = (
    "chat", "streaming", "structured_outputs", "vision", "image_generation",
    "image_editing", "audio_input", "audio_output", "speech_to_text",
    "text_to_speech", "speech_to_speech", "audio_generation", "video_input",
    "video_generation", "image_to_video", "video_understanding", "tool_calling",
    "embeddings", "reasoning", "voice_activity_detection", "audio_translation",
    "video_captioning", "video_summarization", "video_to_video", "audio_to_audio",
)


@dataclass(slots=True)
class CapabilitySnapshot:
    capabilities: dict[str, bool] = field(default_factory=dict)
    evidence: dict[str, list[str]] = field(default_factory=dict)

    def enabled(self, capability: str) -> bool:
        return bool(self.capabilities.get(capability, False))

    def as_dict(self) -> dict[str, Any]:
        return {"capabilities": dict(self.capabilities), "evidence": dict(self.evidence)}


def build_capability_snapshot(
    *,
    base: dict[str, bool] | None = None,
    provider_capabilities: set[str] | frozenset[str] | None = None,
    configured_features: dict[str, bool] | None = None,
) -> CapabilitySnapshot:
    base = base or {}
    provider_capabilities = provider_capabilities or frozenset()
    configured_features = configured_features or {}
    values = {name: False for name in KNOWN_CAPABILITIES}
    evidence: dict[str, list[str]] = {name: [] for name in KNOWN_CAPABILITIES}

    for name, enabled in base.items():
        if name in values and bool(enabled):
            values[name] = True
            evidence[name].append("base_model")
    for name in provider_capabilities:
        if name in values:
            values[name] = True
            evidence[name].append("runtime_provider")
    for name, enabled in configured_features.items():
        if name in values and not enabled:
            values[name] = False
            evidence[name].append("disabled_by_config")

    return CapabilitySnapshot(values, {k: v for k, v in evidence.items() if v})
