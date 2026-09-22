"""Dedicated embedding service contracts for the Gopi platform.

The embedding API is intentionally separate from the generative model.  A
production deployment can inject a neural encoder, while the deterministic
hash encoder remains useful for local development and contract tests.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from evaluation.embeddings import HashEmbeddingModel


class EmbeddingEncoder(Protocol):
    dimension: int

    def encode(self, texts: Sequence[str]) -> torch.Tensor: ...


@dataclass(frozen=True)
class EmbeddingResult:
    embeddings: list[list[float]]
    prompt_tokens: int


class EmbeddingService:
    def __init__(self, encoder: EmbeddingEncoder | None = None) -> None:
        self.encoder = encoder or HashEmbeddingModel(dimension=384)
        dimension = int(getattr(self.encoder, "dimension", 0))
        if dimension < 1:
            raise ValueError("embedding encoder must expose a positive dimension")
        self.dimension = dimension

    def encode(self, texts: Sequence[str], *, normalize: bool = False) -> EmbeddingResult:
        if not texts:
            raise ValueError("input must contain at least one text")
        normalized = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("embedding inputs must be non-empty strings")
            normalized.append(text)
        vectors = self.encoder.encode(normalized)
        if not isinstance(vectors, torch.Tensor) or vectors.ndim != 2:
            raise ValueError("embedding encoder must return a 2D tensor")
        if vectors.shape != (len(normalized), self.dimension):
            raise ValueError("embedding encoder returned an unexpected shape")
        if not torch.isfinite(vectors).all():
            raise ValueError("embedding encoder returned non-finite values")
        vectors = vectors.detach().float()
        if normalize:
            norms = vectors.norm(dim=-1, keepdim=True)
            if torch.any(norms == 0):
                raise ValueError("cannot normalize a zero embedding vector")
            vectors = vectors / norms
        # The dedicated API reports tokenizer-independent service accounting.
        # For an injected encoder, whitespace tokenization is a conservative
        # lower-cost proxy until a tokenizer-aware encoder is configured.
        tokens = sum(max(1, len(text.split())) for text in normalized)
        return EmbeddingResult(vectors.tolist(), tokens)
