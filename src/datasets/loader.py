"""Dataset readers and tokenized causal-language-model dataset."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, Dataset

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
        self.loss_masks: list[torch.Tensor] = []
        for record in records:
            try:
                if isinstance(record, Mapping) and isinstance(record.get("messages"), list):
                    identifiers, loss_mask = self._encode_chat(record["messages"], tokenizer, add_bos, add_eos)
                else:
                    text = record if isinstance(record, str) else record_to_text(record)
                    identifiers = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos, allowed_special="all")
                    loss_mask = [True] * len(identifiers)
            except Exception:
                continue
            identifiers, loss_mask = identifiers[:max_length], loss_mask[:max_length]
            if add_eos and len(identifiers) == max_length:
                eos = tokenizer.token_to_id("<|eos|>")
                if eos is not None:
                    identifiers[-1] = eos
                    loss_mask[-1] = True
            if len(identifiers) >= drop_shorter_than:
                self.examples.append(torch.tensor(identifiers, dtype=torch.long))
                self.loss_masks.append(torch.tensor(loss_mask, dtype=torch.bool))

    @staticmethod
    def _encode_chat(messages, tokenizer: Tokenizer, add_bos: bool, add_eos: bool):
        identifiers: list[int] = []
        mask: list[bool] = []
        if add_bos:
            bos = tokenizer.token_to_id("<|bos|>")
            if bos is None:
                raise ValueError("tokenizer does not define <|bos|>")
            identifiers.append(bos)
            mask.append(False)
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            role = message.get("role")
            content = str(message.get("content", "")).strip()
            if role not in {"system", "user", "assistant"} or not content:
                continue
            piece = tokenizer.encode(f"<|{role}|>\n{content}\n", allowed_special="all")
            identifiers.extend(piece)
            mask.extend([role == "assistant"] * len(piece))
        if not identifiers or (add_bos and len(identifiers) == 1):
            raise ValueError("invalid chat message")
        if add_eos:
            eos = tokenizer.token_to_id("<|eos|>")
            if eos is None:
                raise ValueError("tokenizer does not define <|eos|>")
            identifiers.append(eos)
            mask.append(True)
        return identifiers, mask

    @classmethod
    def from_files(
        cls, paths: Iterable[str | Path], tokenizer: Tokenizer, **kwargs: Any
    ) -> "TextDataset":
        records = (record for path in paths for record in iter_records(path))
        return cls(records, tokenizer, **kwargs)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.examples[index], "loss_mask": self.loss_masks[index]}


class LazyJSONLDataset(Dataset[dict[str, torch.Tensor]]):
    """Index JSONL byte offsets and tokenize records only when requested."""

    def __init__(self, path: str | Path, tokenizer: Tokenizer, *, max_length: int) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.offsets: list[int] = []
        self.lengths: list[int] = []
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
                    self.lengths.append(min(max_length, max(2, len(line) // 4)))

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            line = stream.readline()
            record = json.loads(line) if line.strip() else {}
        dataset = TextDataset([record] if record else [], self.tokenizer, max_length=self.max_length)
        if not dataset:
            # Fallback to adjacent valid index or dummy sequence if corrupted
            fallback_index = (index + 1) % len(self)
            with self.path.open("rb") as stream:
                stream.seek(self.offsets[fallback_index])
                line = stream.readline()
                fallback_record = json.loads(line) if line.strip() else {}
            dataset = TextDataset([fallback_record], self.tokenizer, max_length=self.max_length)
            if not dataset:
                dummy_text = "<|system|>\nYou are Gopi.\n<|user|>\nHi\n<|assistant|>\nHello\n"
                dataset = TextDataset([dummy_text], self.tokenizer, max_length=self.max_length)
        return dataset[0]


def build_text_dataset(paths: Iterable[str | Path], tokenizer: Tokenizer, *, max_length: int, lazy: bool = True):
    datasets = []
    for raw_path in paths:
        path = Path(raw_path)
        if lazy and path.suffix.lower() == ".jsonl":
            datasets.append(LazyJSONLDataset(path, tokenizer, max_length=max_length))
        else:
            datasets.append(TextDataset(iter_records(path), tokenizer, max_length=max_length))
    if not datasets:
        raise ValueError("no dataset paths configured")
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    lengths: list[int] = []
    for item in datasets:
        lengths.extend(item.lengths if hasattr(item, "lengths") else [example.numel() for example in item.examples])
    dataset.lengths = lengths
    return dataset
