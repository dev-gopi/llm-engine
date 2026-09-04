"""Dependency-light image-folder datasets for vision and diffusion training."""

from __future__ import annotations

from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset

from .processor import ImageProcessor


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


def discover_images(root: str | Path) -> list[Path]:
    directory = Path(root)
    if not directory.is_dir():
        raise FileNotFoundError(f"image directory not found: {directory}")
    images = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"no supported images found under: {directory}")
    return images


class ImageDataset(Dataset[Tensor]):
    def __init__(self, root: str | Path, image_size: int, *, augment: bool = False,
                 processor: ImageProcessor | None = None) -> None:
        self.paths = discover_images(root)
        self.image_size = image_size
        self.processor = processor or ImageProcessor(
            image_size, resize_mode="random_crop" if augment else "center_crop", augment=augment
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        return self.processor(self.paths[index])


class ImageClassificationDataset(Dataset[tuple[Tensor, int]]):
    """Read ``root/class_name/image`` folders with stable alphabetical labels."""

    def __init__(self, root: str | Path, image_size: int, *, augment: bool = False,
                 processor: ImageProcessor | None = None) -> None:
        root_path = Path(root)
        classes = sorted(path.name for path in root_path.iterdir() if path.is_dir()) if root_path.is_dir() else []
        if len(classes) < 2:
            raise ValueError("classification data needs at least two class directories")
        self.class_to_id = {name: index for index, name in enumerate(classes)}
        self.samples = [
            (path, self.class_to_id[path.relative_to(root_path).parts[0]])
            for path in discover_images(root_path)
            if path.relative_to(root_path).parts[0] in self.class_to_id
        ]
        self.image_size = image_size
        self.processor = processor or ImageProcessor(
            image_size, resize_mode="random_crop" if augment else "center_crop", augment=augment
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        path, label = self.samples[index]
        return self.processor(path), label
