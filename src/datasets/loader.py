"""Dataset readers and tokenized causal-language-model dataset."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from tokenizer.encoder import Tokenizer

from .preprocessor import record_to_text


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"dataset not found: {source}")
    with source.open(encoding="utf-8") as stream:
        if source.suffix.lower() == ".jsonl":
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {source}:{line_number}") from error
                if not isinstance(record, dict):
                    raise ValueError(f"record at {source}:{line_number} must be an object")
                yield record
        elif source.suffix.lower() == ".json":
            records = json.load(stream)
            if not isinstance(records, list):
                raise ValueError("JSON dataset root must be a list")
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError("JSON dataset records must be objects")
                yield record
        else:
            raise ValueError("dataset must be .json or .jsonl")


class TextDataset(Dataset[dict[str, torch.Tensor]]):
    """Tokenize records into bounded causal-LM sequences."""

    def __init__(
        self,
        records: Iterable[str | Mapping[str, Any]],
        tokenizer: Tokenizer,
        *,
        max_length: int,
        add_bos: bool = True,
        add_eos: bool = True,
        drop_shorter_than: int = 2,
    ) -> None:
        if max_length < 2:
            raise ValueError("max_length must be at least two")
        self.examples: list[torch.Tensor] = []
        for record in records:
            text = record if isinstance(record, str) else record_to_text(record)
            identifiers = tokenizer.encode(
                text, add_bos=add_bos, add_eos=add_eos, allowed_special="all"
            )[:max_length]
            if len(identifiers) >= drop_shorter_than:
                self.examples.append(torch.tensor(identifiers, dtype=torch.long))

    @classmethod
    def from_files(
        cls, paths: Iterable[str | Path], tokenizer: Tokenizer, **kwargs: Any
    ) -> "TextDataset":
        records = (record for path in paths for record in iter_records(path))
        return cls(records, tokenizer, **kwargs)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.examples[index]}
