"""Safe WAV I/O helpers without mandatory native audio dependencies."""

from __future__ import annotations

import random
import wave
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


def load_wav(
    path: str | Path,
    *,
    sample_rate: int,
    samples: int | None = None,
    crop: str = "start",
    peak_normalize: bool = False,
) -> Tensor:
    source = Path(path)
    if crop not in {"start", "center", "random"}:
        raise ValueError("crop must be start, center, or random")
    with wave.open(str(source), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError("built-in WAV loader supports 16-bit PCM; install/convert data accordingly")
        channels = handle.getnchannels()
        source_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    data = data.reshape(-1, channels).mean(axis=1)
    tensor = torch.from_numpy(data)
    if source_rate != sample_rate:
        target = max(1, round(tensor.numel() * sample_rate / source_rate))
        tensor = F.interpolate(tensor[None, None], size=target, mode="linear", align_corners=False)[0, 0]
    if peak_normalize and tensor.numel():
        peak = tensor.abs().max()
        if peak > 1e-6:
            tensor = tensor / peak.clamp_min(1e-6) * 0.98
    if samples is not None:
        if samples <= 0:
            raise ValueError("samples must be positive")
        if tensor.numel() < samples:
            tensor = F.pad(tensor, (0, samples - tensor.numel()))
        elif tensor.numel() > samples:
            available = tensor.numel() - samples
            if crop == "center":
                start = available // 2
            elif crop == "random":
                start = random.randint(0, available)
            else:
                start = 0
            tensor = tensor[start:start + samples]
        else:
            tensor = tensor[:samples]
    return tensor


def save_wav(path: str | Path, waveform: Tensor, sample_rate: int) -> Path:
    if waveform.ndim != 1:
        raise ValueError("waveform must be mono with shape [samples]")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pcm = (waveform.detach().float().cpu().clamp(-1, 1).numpy() * 32767.0).round().astype("<i2")
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return destination
