"""Image-text SFT dataset and projector evaluation helpers."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .model import (
    ImageTextSFTExample,
    VisionLanguageModel,
    multimodal_sft_metrics,
)


class ImageTextSFTDataset(Dataset[ImageTextSFTExample]):
    """Lazy JSONL image-text dataset for frozen-backbone projector SFT."""

    def __init__(self, manifest: str | Path, tokenizer, *, image_size: int = 128, transform: Callable | None = None) -> None:
        self.manifest = Path(manifest)
        self.tokenizer = tokenizer
        self.image_size = int(image_size)
        self.transform = transform
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        self.records: list[dict] = []
        with self.manifest.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict) or not record.get("image") or not record.get("response"):
                    raise ValueError(f"invalid image-text record at line {line_number}")
                self.records.append(record)
        if not self.records:
            raise ValueError("image-text SFT dataset is empty")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ImageTextSFTExample:
        record = self.records[index]
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for image-text SFT data") from exc
        image_path = (self.manifest.parent / record["image"]).resolve()
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize((self.image_size, self.image_size))
            tensor = torch.frombuffer(image.tobytes(), dtype=torch.uint8).reshape(self.image_size, self.image_size, 3).permute(2, 0, 1).float() / 255.0
        if self.transform is not None:
            tensor = self.transform(tensor)
        prompt = str(record.get("prompt", "Describe the image."))
        prompt_ids = torch.tensor(self.tokenizer.encode(prompt, add_bos=True), dtype=torch.long)
        response_ids = torch.tensor(self.tokenizer.encode(str(record["response"]), add_bos=False), dtype=torch.long)
        return ImageTextSFTExample(tensor, prompt_ids, response_ids, str(record.get("id", index)))


def evaluate_projector_batch(model: VisionLanguageModel, batch: dict[str, Tensor]) -> dict[str, float]:
    """Evaluate response-only token accuracy/perplexity for one batch."""
    model.eval()
    with torch.inference_mode():
        logits, _ = model(batch["images"], batch["prompt_ids"], batch["response_ids"])
    return multimodal_sft_metrics(logits, batch["response_ids"], batch["response_loss_mask"])
