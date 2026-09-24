"""Lightweight model-independent quality diagnostics for generated media."""
from __future__ import annotations
from torch import Tensor


def audio_diagnostics(waveform: Tensor) -> dict[str, float]:
    x = waveform.detach().float().flatten()
    if x.numel() == 0:
        raise ValueError("waveform is empty")
    peak = x.abs().max().item()
    rms = x.square().mean().sqrt().item()
    clipping = (x.abs() >= 0.999).float().mean().item()
    dc = x.mean().item()
    return {"peak": peak, "rms": rms, "clipping_fraction": clipping, "dc_offset": dc}


def video_diagnostics(video: Tensor) -> dict[str, float]:
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError("video must have shape [3, frames, height, width]")
    x = video.detach().float()
    temporal = (x[:, 1:] - x[:, :-1]).abs().mean().item() if x.shape[1] > 1 else 0.0
    saturation = (x.abs() >= 0.999).float().mean().item()
    return {"temporal_abs_delta": temporal, "saturation_fraction": saturation, "mean": x.mean().item(), "std": x.std().item()}
