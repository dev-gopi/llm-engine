"""Image dataset integrity and duplicate inspection."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from .dataset import discover_images
from .processor import _pillow_modules


@dataclass(frozen=True)
class ImageAudit:
    images: int
    readable: int
    corrupt: int
    exact_duplicates: int
    min_width: int
    min_height: int
    max_width: int
    max_height: int
    total_bytes: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def audit_images(root: str | Path) -> ImageAudit:
    """Decode every image and count byte-identical files without modifying data."""
    image_module, _, _ = _pillow_modules()
    paths = discover_images(root)
    readable = corrupt = duplicates = total_bytes = 0
    widths: list[int] = []
    heights: list[int] = []
    digests: set[str] = set()
    for path in paths:
        try:
            content = path.read_bytes()
            total_bytes += len(content)
            digest = hashlib.sha256(content).hexdigest()
            if digest in digests:
                duplicates += 1
            digests.add(digest)
            with image_module.open(path) as image:
                image.load()
                width, height = image.size
            widths.append(width)
            heights.append(height)
            readable += 1
        except (OSError, ValueError):
            corrupt += 1
    return ImageAudit(
        images=len(paths), readable=readable, corrupt=corrupt,
        exact_duplicates=duplicates,
        min_width=min(widths, default=0), min_height=min(heights, default=0),
        max_width=max(widths, default=0), max_height=max(heights, default=0),
        total_bytes=total_bytes,
    )
