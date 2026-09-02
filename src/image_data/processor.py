"""Pillow-based image processing without a torchvision dependency."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


def _pillow_image() -> Any:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("image support requires: pip install -e '.[images]'") from error
    return Image


def load_image(path: str | Path, image_size: int) -> Tensor:
    """Load an RGB image as a normalized [3, H, W] float tensor in [-1, 1]."""
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    image_module = _pillow_image()
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"image not found: {source}")
    with image_module.open(source) as image:
        image = image.convert("RGB").resize((image_size, image_size))
        array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div(127.5).sub(1.0)


def tensor_to_image(tensor: Tensor) -> Any:
    """Convert one normalized [3, H, W] tensor in [-1, 1] to a Pillow image."""
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("tensor must have shape [3, height, width]")
    image_module = _pillow_image()
    array = tensor.detach().float().clamp(-1, 1).add(1).mul(127.5)
    array = array.round().byte().permute(1, 2, 0).cpu().numpy()
    return image_module.fromarray(array, mode="RGB")
