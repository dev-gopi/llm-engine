"""Dataset readers and tokenized causal-language-model dataset."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, Dataset

from tokenizer.encoder import Tokenizer

from .preprocessor import clean, record_to_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MixtureSource:
    """Versioned, governed input declaration for deterministic streaming."""

    name: str
    domain: str
    version: str
    paths: tuple[Path, ...]
    weight: float
    license_identifier: str
    quality_metrics: Mapping[str, float]


def load_mixture_sources(config: Mapping[str, Any]) -> tuple[MixtureSource, ...]:
    """Validate declared sources without opening or materializing their records."""
    entries = config.get("dataset_mixture", ())
    if not isinstance(entries, list) or not entries:
        raise ValueError("dataset_mixture must be a non-empty list")
    names: set[str] = set()
    sources: list[MixtureSource] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"dataset_mixture[{index}] must be a mapping")
        required = ("name", "domain", "version", "paths", "weight", "license", "quality_metrics")
        missing = [key for key in required if key not in entry]
        if missing:
            raise ValueError(f"dataset_mixture[{index}] missing {', '.join(missing)}")
        name = entry["name"]
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("mixture source names must be unique, non-empty text")
        domain = entry["domain"]
        version = entry["version"]
        paths = entry["paths"]
        license_identifier = entry["license"]
        quality_metrics = entry["quality_metrics"]
        if not isinstance(domain, str) or not isinstance(version, str) or not version.strip():
            raise ValueError("mixture source domain and version must be non-empty text")
        if not isinstance(paths, list) or not paths or not all(isinstance(path, str) and path for path in paths):
            raise ValueError("mixture source paths must be a non-empty list of paths")
        if not isinstance(entry["weight"], (int, float)) or isinstance(entry["weight"], bool) or entry["weight"] <= 0:
            raise ValueError("mixture source weight must be positive")
        if not isinstance(license_identifier, str) or not license_identifier.strip():
            raise ValueError("mixture source license must be non-empty text")
        if not isinstance(quality_metrics, Mapping) or not quality_metrics:
            raise ValueError("mixture source quality_metrics must be a non-empty mapping")
        metrics = {}
        for metric, value in quality_metrics.items():
            if not isinstance(metric, str) or not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError("mixture quality metrics must map names to numbers")
            metrics[metric] = float(value)
        names.add(name)
        sources.append(MixtureSource(
            name=name, domain=domain, version=version, paths=tuple(Path(path) for path in paths),
            weight=float(entry["weight"]), license_identifier=license_identifier,
            quality_metrics=metrics,
        ))
    return tuple(sources)


def iter_mixture_records(sources: tuple[MixtureSource, ...]) -> Iterator[dict[str, Any]]:
    """Yield a finite, weighted round-robin stream annotated with source identity.

    A source is opened only through :func:`iter_records`; no corpus is loaded
    into memory.  Ties are resolved by declaration order for reproducibility.
    """
    streams = [iter(record for path in source.paths for record in iter_records(path)) for source in sources]
    credits = [0.0] * len(sources)
    active = set(range(len(sources)))
    total_weight = sum(source.weight for source in sources)
    while active:
        for index in active:
            credits[index] += sources[index].weight
        selected = max(active, key=lambda index: (credits[index], -index))
        try:
            record = next(streams[selected])
        except StopIteration:
            active.remove(selected)
            continue
        credits[selected] -= total_weight
        yield {**record, "_mixture_source": sources[selected].name,
               "_mixture_version": sources[selected].version}

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


def _packed_identifiers(record, tokenizer, max_length, add_bos, fingerprint):
    if record.get("tokenizer_fingerprint") != fingerprint:
        raise ValueError("packed record tokenizer fingerprint does not match")
    packed_ids = record["token_ids"]
    if not isinstance(packed_ids, list) or not packed_ids or any(
        not isinstance(value, int) or isinstance(value, bool)
        or not 0 <= value < tokenizer.vocab_size for value in packed_ids
    ):
        raise ValueError("packed token_ids must be a non-empty list of valid token IDs")
    identifiers = list(packed_ids)
    if add_bos:
        bos = tokenizer.token_to_id("<|bos|>")
        if bos is None:
            raise ValueError("tokenizer does not define <|bos|>")
        identifiers.insert(0, bos)
    if len(identifiers) > max_length:
        raise ValueError("packed record exceeds max_length; repack with the selected context length")
    return identifiers


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
        packed_fingerprint = None
        for record_index, record in enumerate(records):
            try:
                if isinstance(record, Mapping) and record.get("prepacked") is True and "token_ids" in record:
                    if packed_fingerprint is None:
                        packed_fingerprint = tokenizer.fingerprint
                    identifiers = _packed_identifiers(record, tokenizer, max_length, add_bos, packed_fingerprint)
                    loss_mask = [True] * len(identifiers)
                elif isinstance(record, Mapping) and isinstance(record.get("messages"), list):
                    identifiers, loss_mask = self._encode_chat(record["messages"], tokenizer, add_bos, add_eos)
                elif (
                    isinstance(record, Mapping)
                    and isinstance(record.get("prompt"), str)
                    and isinstance(record.get("chosen"), str)
                ):
                    # Preference corpora contribute the selected response to SFT;
                    # the rejected response must never become a training target.
                    messages = (
                        {"role": "user", "content": record["prompt"]},
                        {"role": "assistant", "content": record["chosen"]},
                    )
                    identifiers, loss_mask = self._encode_chat(messages, tokenizer, add_bos, add_eos)
                else:
                    text = record if isinstance(record, str) else record_to_text(record)
                    prepacked = isinstance(record, Mapping) and record.get("prepacked") is True
                    identifiers = tokenizer.encode(
                        text, add_bos=add_bos, add_eos=add_eos and not prepacked,
                        allowed_special="all",
                    )
                    loss_mask = [True] * len(identifiers)
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid dataset record at index {record_index}: {error}") from error
            # Keep the last retained content token when an example is too long.
            # Injecting EOS at this boundary would teach the model that an
            # incomplete text or assistant response is a valid stopping point.
            identifiers, loss_mask = identifiers[:max_length], loss_mask[:max_length]
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
            content = clean(str(message.get("content", "")))
            if role not in {"system", "user", "assistant"} or not content:
                continue
            content = content.replace("<|", "< |")
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
        self._packed_fingerprint = None
        self.offsets: list[int] = []
        self.line_numbers: list[int] = []
        self.lengths: list[int] = []
        with self.path.open("rb") as stream:
            line_number = 0
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                line_number += 1
                if line.strip():
                    self.offsets.append(offset)
                    self.line_numbers.append(line_number)
                    self.lengths.append(min(max_length, max(2, len(line) // 4)))

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            line = stream.readline()
        location = f"{self.path}:{self.line_numbers[index]}"
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid JSON at {location}") from error
        if not isinstance(record, dict):
            raise ValueError(f"record at {location} must be an object")
        try:
            if record.get("prepacked") is True and "token_ids" in record:
                if self._packed_fingerprint is None:
                    self._packed_fingerprint = self.tokenizer.fingerprint
                identifiers = _packed_identifiers(record, self.tokenizer, self.max_length,
                                                  True, self._packed_fingerprint)
                return {"input_ids": torch.tensor(identifiers, dtype=torch.long),
                        "loss_mask": torch.ones(len(identifiers), dtype=torch.bool)}
            dataset = TextDataset([record], self.tokenizer, max_length=self.max_length)
        except (TypeError, ValueError) as error:
            raise ValueError(f"unusable dataset record at {location}: {error}") from error
        if not dataset:
            raise ValueError(f"unusable dataset record at {location}: token sequence is too short")
        return dataset[0]


def build_text_dataset(paths: Iterable[str | Path], tokenizer: Tokenizer, *, max_length: int, lazy: bool = True):
    datasets = []
    for raw_path in paths:
        path = Path(raw_path)
        logger.info("Preparing dataset %s (lazy=%s)", path, lazy)
        if lazy and path.suffix.lower() == ".jsonl":
            datasets.append(LazyJSONLDataset(path, tokenizer, max_length=max_length))
        else:
            datasets.append(TextDataset(iter_records(path), tokenizer, max_length=max_length))
        logger.info("Dataset ready: %s (%d examples)", path, len(datasets[-1]))
    if not datasets:
        raise ValueError("no dataset paths configured")
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    lengths: list[int] = []
    dataset_sizes: list[int] = []
    for item in datasets:
        dataset_sizes.append(len(item))
        lengths.extend(item.lengths if hasattr(item, "lengths") else [example.numel() for example in item.examples])
    dataset.lengths = lengths
    dataset.dataset_sizes = dataset_sizes
    return dataset
