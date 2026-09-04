"""Configurable Pillow image processing without a torchvision dependency."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor


def _pillow_modules() -> tuple[Any, Any, Any]:
    try:
        from PIL import Image, ImageEnhance, ImageOps
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("image support requires: pip install -e '.[images]'") from error
    return Image, ImageEnhance, ImageOps


@dataclass(frozen=True)
class ImageProcessor:
    """Decode, resize, augment, and normalize images consistently."""

    image_size: int
    resize_mode: str = "center_crop"
    normalization: str = "minus_one_one"
    augment: bool = False
    horizontal_flip_probability: float = 0.5
    color_jitter: float = 0.0

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, training: bool = False) -> "ImageProcessor":
        flip_probability = float(config.get("horizontal_flip_probability", 0.5))
        if not bool(config.get("horizontal_flip", True)):
            flip_probability = 0.0
        return cls(
            image_size=int(config["image_size"]),
            resize_mode=str(config.get(
                "train_resize_mode" if training else "eval_resize_mode",
                "random_crop" if training else "center_crop",
            )),
            normalization=str(config.get("image_normalization", "minus_one_one")),
            augment=training,
            horizontal_flip_probability=flip_probability,
            color_jitter=float(config.get("color_jitter", 0.0)),
        )

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.resize_mode not in {"stretch", "center_crop", "random_crop"}:
            raise ValueError("resize_mode must be stretch, center_crop, or random_crop")
        if self.normalization not in {"minus_one_one", "zero_one", "imagenet"}:
            raise ValueError("normalization must be minus_one_one, zero_one, or imagenet")
        if not 0 <= self.horizontal_flip_probability <= 1:
            raise ValueError("horizontal_flip_probability must be between zero and one")
        if not 0 <= self.color_jitter <= 1:
            raise ValueError("color_jitter must be between zero and one")

    def __call__(self, path: str | Path) -> Tensor:
        image_module, enhance_module, ops_module = _pillow_modules()
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"image not found: {source}")
        with image_module.open(source) as opened:
            image = ops_module.exif_transpose(opened).convert("RGB")
            image = self._resize(image, image_module)
            if self.augment and torch.rand(()) < self.horizontal_flip_probability:
                image = ops_module.mirror(image)
            if self.augment and self.color_jitter:
                brightness = 1 + (float(torch.rand(())) * 2 - 1) * self.color_jitter
                contrast = 1 + (float(torch.rand(())) * 2 - 1) * self.color_jitter
                image = enhance_module.Brightness(image).enhance(brightness)
                image = enhance_module.Contrast(image).enhance(contrast)
            array = np.asarray(image, dtype=np.float32).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).div(255.0)
        if self.normalization == "minus_one_one":
            return tensor.mul(2).sub(1)
        if self.normalization == "imagenet":
            mean = tensor.new_tensor((0.485, 0.456, 0.406))[:, None, None]
            std = tensor.new_tensor((0.229, 0.224, 0.225))[:, None, None]
            return (tensor - mean) / std
        return tensor

    def _resize(self, image: Any, image_module: Any) -> Any:
        resampling = image_module.Resampling.BICUBIC
        if self.resize_mode == "stretch":
            return image.resize((self.image_size, self.image_size), resampling)
        width, height = image.size
        scale = self.image_size / min(width, height)
        resized = image.resize(
            (max(self.image_size, round(width * scale)),
             max(self.image_size, round(height * scale))), resampling
        )
        left_max = resized.width - self.image_size
        top_max = resized.height - self.image_size
        if self.resize_mode == "random_crop" and self.augment:
            left = int(torch.randint(left_max + 1, (1,))) if left_max else 0
            top = int(torch.randint(top_max + 1, (1,))) if top_max else 0
        else:
            left, top = left_max // 2, top_max // 2
        return resized.crop((left, top, left + self.image_size, top + self.image_size))


def load_image(path: str | Path, image_size: int) -> Tensor:
    """Backward-compatible RGB loading into ``[-1, 1]`` using square resize."""
    return ImageProcessor(image_size, resize_mode="stretch")(path)


def tensor_to_image(tensor: Tensor) -> Any:
    """Convert a normalized ``[3, H, W]`` tensor in ``[-1, 1]`` to Pillow."""
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("tensor must have shape [3, height, width]")
    image_module, _, _ = _pillow_modules()
    array = tensor.detach().float().clamp(-1, 1).add(1).mul(127.5)
    array = array.round().byte().permute(1, 2, 0).cpu().numpy()
    return image_module.fromarray(array, mode="RGB")
