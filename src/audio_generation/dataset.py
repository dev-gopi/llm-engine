"""JSONL caption/audio dataset for text-to-audio training."""

from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset

from .io import load_wav


class AudioCaptionDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        *,
        sample_rate: int,
        duration_seconds: float,
        crop: str = "start",
        peak_normalize: bool = False,
        validate_files: bool = True,
    ) -> None:
        self.manifest = Path(manifest)
        if not self.manifest.is_file():
            raise FileNotFoundError(f"audio manifest not found: {self.manifest}")
        if sample_rate <= 0 or duration_seconds <= 0:
            raise ValueError("sample_rate and duration_seconds must be positive")
        self.sample_rate = sample_rate
        self.samples = round(sample_rate * duration_seconds)
        self.crop = crop
        self.peak_normalize = peak_normalize
        self.records: list[dict[str, str]] = []
        with self.manifest.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record.get("audio"), str) or not isinstance(record.get("text"), str):
                    raise ValueError(f"invalid audio manifest record at line {line_number}")
                if not record["text"].strip():
                    raise ValueError(f"empty audio caption at line {line_number}")
                path = Path(record["audio"])
                if not path.is_absolute():
                    path = self.manifest.parent / path
                if validate_files and not path.is_file():
                    raise FileNotFoundError(f"audio file not found at line {line_number}: {path}")
                self.records.append(record)
        if not self.records:
            raise ValueError("audio manifest is empty")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        path = Path(record["audio"])
        if not path.is_absolute():
            path = self.manifest.parent / path
        return load_wav(
            path,
            sample_rate=self.sample_rate,
            samples=self.samples,
            crop=self.crop,
            peak_normalize=self.peak_normalize,
        ), record["text"]
