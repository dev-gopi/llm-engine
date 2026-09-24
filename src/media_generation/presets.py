"""Inference presets and prompt helpers for production media generation."""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

@dataclass(frozen=True)
class InferencePreset:
    name: str
    steps: int
    guidance_scale: float
    eta: float = 0.0

DEFAULT_PRESETS: dict[str, InferencePreset] = {
    "draft": InferencePreset("draft", 20, 3.5, 0.0),
    "balanced": InferencePreset("balanced", 40, 4.5, 0.0),
    "quality": InferencePreset("quality", 70, 5.5, 0.0),
    "max_quality": InferencePreset("max_quality", 100, 6.0, 0.0),
}

STYLE_SUFFIXES: dict[str, str] = {
    "cinematic": "cinematic composition, coherent detail, professional lighting, polished production quality",
    "photoreal": "photorealistic, physically plausible detail, natural lighting, high dynamic range",
    "anime": "clean anime aesthetic, expressive composition, consistent line work and color design",
    "ambient": "spacious ambient texture, clean mix, controlled dynamics, immersive atmosphere",
    "music": "professional music production, coherent structure, balanced mix, clean transients",
    "speech": "clear intelligible speech, natural prosody, low background noise, studio-quality recording",
}


def resolve_preset(name: str | None, config: Mapping[str, Any]) -> InferencePreset:
    if not name:
        return InferencePreset(
            "config",
            int(config.get("inference_steps", 50)),
            float(config.get("guidance_scale", 4.5)),
            float(config.get("ddim_eta", 0.0)),
        )
    key = name.strip().lower()
    custom = config.get("inference_presets", {}) or {}
    if key in custom:
        values = custom[key]
        return InferencePreset(key, int(values["steps"]), float(values["guidance_scale"]), float(values.get("eta", 0.0)))
    if key not in DEFAULT_PRESETS:
        raise ValueError(f"unknown inference preset: {name}")
    return DEFAULT_PRESETS[key]


def apply_style(prompt: str, style: str | None) -> str:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt cannot be empty")
    if not style:
        return prompt
    key = style.strip().lower()
    suffix = STYLE_SUFFIXES.get(key)
    if suffix is None:
        raise ValueError(f"unknown style preset: {style}")
    return f"{prompt}, {suffix}"
