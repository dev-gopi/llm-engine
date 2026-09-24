"""JSONL caption/video dataset for text-to-video training."""

from __future__ import annotations

import json
import random
from pathlib import Path

from torch.utils.data import Dataset

from .io import load_video


class VideoCaptionDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        *,
        frames: int,
        height: int,
        width: int,
        sampling: str = "uniform",
        horizontal_flip_probability: float = 0.0,
        validate_files: bool = True,
    ) -> None:
        self.manifest = Path(manifest)
        if not self.manifest.is_file():
            raise FileNotFoundError(f"video manifest not found: {self.manifest}")
        if min(frames, height, width) <= 0:
            raise ValueError("frames, height, and width must be positive")
        if not 0 <= horizontal_flip_probability <= 1:
            raise ValueError("horizontal_flip_probability must be between zero and one")
        self.frames, self.height, self.width = frames, height, width
        self.sampling = sampling
        self.horizontal_flip_probability = horizontal_flip_probability
        self.records: list[dict[str, str]] = []
        with self.manifest.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record.get("video"), str) or not isinstance(record.get("text"), str):
                    raise ValueError(f"invalid video manifest record at line {line_number}")
                if not record["text"].strip():
                    raise ValueError(f"empty video caption at line {line_number}")
                path = Path(record["video"])
                if not path.is_absolute():
                    path = self.manifest.parent / path
                if validate_files and not path.is_file():
                    raise FileNotFoundError(f"video file not found at line {line_number}: {path}")
                self.records.append(record)
        if not self.records:
            raise ValueError("video manifest is empty")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        path = Path(record["video"])
        if not path.is_absolute():
            path = self.manifest.parent / path
        flip = self.horizontal_flip_probability > 0 and random.random() < self.horizontal_flip_probability
        return load_video(
            path,
            frames=self.frames,
            height=self.height,
            width=self.width,
            sampling=self.sampling,
            horizontal_flip=flip,
        ), record["text"]
