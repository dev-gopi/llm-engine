"""Device selection helpers."""

import torch


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    if isinstance(requested, torch.device):
        device = requested
    elif requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def get_device() -> str:
    """Backward-compatible device name helper."""
    return str(resolve_device())
