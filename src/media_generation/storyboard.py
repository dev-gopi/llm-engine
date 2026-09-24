"""Storyboard and production prompt utilities for multi-shot video generation."""
from __future__ import annotations

from dataclasses import dataclass

CAMERA_SUFFIXES = {
    "static": "locked-off camera, stable framing",
    "dolly_in": "slow cinematic dolly-in, stable camera motion",
    "dolly_out": "slow cinematic dolly-out, stable camera motion",
    "pan_left": "smooth camera pan to the left",
    "pan_right": "smooth camera pan to the right",
    "orbit": "smooth cinematic orbit around the subject",
    "handheld": "controlled handheld camera movement, natural micro-motion",
    "drone": "smooth aerial drone movement, wide establishing perspective",
}


@dataclass(frozen=True)
class StoryboardShot:
    index: int
    prompt: str
    camera: str | None = None


def build_storyboard(base_prompt: str, scenes: list[str] | None = None, *, camera: str | None = None) -> list[StoryboardShot]:
    base = base_prompt.strip()
    if not base:
        raise ValueError("base prompt cannot be empty")
    items = [s.strip() for s in (scenes or []) if s and s.strip()] or [base]
    if camera is not None and camera not in CAMERA_SUFFIXES:
        raise ValueError(f"unknown camera preset: {camera}")
    suffix = CAMERA_SUFFIXES.get(camera)
    shots=[]
    for index, text in enumerate(items):
        prompt = text if text != base else base
        if suffix:
            prompt = f"{prompt}, {suffix}"
        shots.append(StoryboardShot(index=index, prompt=prompt, camera=camera))
    return shots
