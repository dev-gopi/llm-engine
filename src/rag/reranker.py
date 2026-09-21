"""Cross-encoder/reranker interface with deterministic local baseline."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Sequence
import re

@dataclass(frozen=True)
class RerankedDocument:
    document: object
    score: float
    rank: int

class Reranker(Protocol):
    def rerank(self, query: str, documents: Sequence[object], *, top_k: int = 5) -> list[RerankedDocument]: ...

class LexicalCrossEncoderBaseline:
    """Dependency-free deterministic baseline; replace with a trained cross-encoder without changing the interface."""
    def rerank(self, query: str, documents: Sequence[object], *, top_k: int = 5) -> list[RerankedDocument]:
        if top_k < 1: raise ValueError("top_k must be positive")
        q = set(re.findall(r"\w+", query.casefold()))
        scored = []
        for i, doc in enumerate(documents):
            text = getattr(doc, "description", None) or getattr(doc, "text", None) or str(doc)
            words = set(re.findall(r"\w+", str(text).casefold()))
            score = len(q & words) / max(len(q), 1)
            scored.append((score, i, doc))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [RerankedDocument(doc, score, rank) for rank, (score, _, doc) in enumerate(scored[:top_k], 1)]
