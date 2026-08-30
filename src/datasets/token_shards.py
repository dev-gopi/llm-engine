"""Memory-mapped-friendly binary token shard dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenShardDataset(Dataset[dict[str, torch.Tensor]]):
    """Read fixed-width uint32 token sequences from NumPy shards on demand."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "gopi-token-shards-v1":
            raise ValueError("unsupported token shard manifest format")
        dtype_name = manifest.get("dtype")
        if dtype_name not in {"uint16", "uint32"}:
            raise ValueError("token shard dtype must be uint16 or uint32")
        self.dtype = np.dtype(dtype_name)
        self.sequence_length = int(manifest["sequence_length"])
        if self.sequence_length < 2:
            raise ValueError("token shard sequence_length must be at least two")
        self.tokenizer_vocab_size = int(manifest.get("tokenizer_vocab_size", 0))
        self.tokenizer_fingerprint = manifest.get("tokenizer_fingerprint")
        if self.tokenizer_fingerprint is not None and not isinstance(
            self.tokenizer_fingerprint, str
        ):
            raise ValueError("token shard tokenizer_fingerprint must be a string")
        self.shards: list[tuple[Path, Path | None, int]] = []
        self.cumulative: list[int] = []
        total = 0
        for item in manifest["shards"]:
            count = int(item["sequences"])
            path = self.manifest_path.parent / item["file"]
            if count < 1:
                raise ValueError("token shard sequence counts must be positive")
            if not path.is_file():
                raise FileNotFoundError(f"token shard not found: {path}")
            expected_bytes = count * self.sequence_length * self.dtype.itemsize
            if path.stat().st_size != expected_bytes:
                raise ValueError(
                    f"token shard size mismatch for {path}: expected {expected_bytes} bytes, "
                    f"found {path.stat().st_size}"
                )
            mask_path = None
            if item.get("loss_mask_file"):
                mask_path = self.manifest_path.parent / item["loss_mask_file"]
                expected_mask_bytes = count * self.sequence_length
                if not mask_path.is_file():
                    raise FileNotFoundError(f"token shard loss mask not found: {mask_path}")
                if mask_path.stat().st_size != expected_mask_bytes:
                    raise ValueError(
                        f"token shard loss-mask size mismatch for {mask_path}: "
                        f"expected {expected_mask_bytes} bytes, found {mask_path.stat().st_size}"
                    )
            self.shards.append((path, mask_path, count))
            total += count
            self.cumulative.append(total)
        self.lengths = [self.sequence_length] * total
        self._arrays: dict[Path, np.memmap] = {}
        self._mask_arrays: dict[Path, np.memmap] = {}

    def __len__(self) -> int:
        return self.cumulative[-1] if self.cumulative else 0

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        import bisect
        shard_index = bisect.bisect_right(self.cumulative, index)
        previous = self.cumulative[shard_index - 1] if shard_index else 0
        path, mask_path, count = self.shards[shard_index]
        array = self._arrays.get(path)
        if array is None:
            array = np.memmap(path, mode="r", dtype=self.dtype, shape=(count, self.sequence_length))
            self._arrays[path] = array
        tokens = torch.from_numpy(np.asarray(array[index - previous]).astype(np.int64, copy=True))
        if mask_path is None:
            loss_mask = torch.ones_like(tokens, dtype=torch.bool)
        else:
            mask_array = self._mask_arrays.get(mask_path)
            if mask_array is None:
                mask_array = np.memmap(
                    mask_path, mode="r", dtype=np.uint8,
                    shape=(count, self.sequence_length),
                )
                self._mask_arrays[mask_path] = mask_array
            loss_mask = torch.from_numpy(
                np.asarray(mask_array[index - previous]).astype(bool, copy=True)
            )
        return {"input_ids": tokens, "loss_mask": loss_mask}
