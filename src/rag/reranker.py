"""Cross-encoder/reranker contracts and deterministic evaluation utilities."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence


@dataclass(frozen=True)
class RerankedDocument:
    """A ranked document; the original object is retained byte-for-byte."""

    document: object
    score: float
    rank: int

    @property
    def citation(self) -> str | None:
        return getattr(self.document, "url", None) or getattr(self.document, "source", None)


class Reranker(Protocol):
    def rerank(
        self, query: str, documents: Sequence[object], *, top_k: int = 5
    ) -> list[RerankedDocument]: ...


def _document_text(document: object) -> str:
    return str(
        getattr(document, "description", None)
        or getattr(document, "text", None)
        or getattr(document, "content", None)
        or document
    )


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


class LexicalCrossEncoderBaseline:
    """Deterministic dependency-free baseline for regression tests.

    This is intentionally called a *baseline*, not a neural cross-encoder. It
    provides a stable ranking contract when an external reranker is unavailable.
    """

    def rerank(self, query: str, documents: Sequence[object], *, top_k: int = 5) -> list[RerankedDocument]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_terms = _tokens(query)
        if not query_terms:
            return []
        q = set(query_terms)
        scored: list[tuple[float, int, object]] = []
        for index, doc in enumerate(documents):
            text = _document_text(doc)
            terms = _tokens(text)
            if not terms:
                score = 0.0
            else:
                counts = {term: terms.count(term) for term in q}
                overlap = sum(1 for term in q if counts[term]) / len(q)
                phrase = 1.0 if query.casefold().strip() in text.casefold() else 0.0
                density = sum(min(counts[term], 3) for term in q) / max(len(terms), 1)
                score = overlap + 0.15 * phrase + 0.10 * density
            scored.append((score, index, doc))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            RerankedDocument(document=doc, score=float(score), rank=rank)
            for rank, (score, _, doc) in enumerate(scored[:top_k], 1)
        ]


class CallableCrossEncoderReranker:
    """Adapter for any cross-encoder scorer.

    `score_pairs` receives ``[(query, document_text), ...]`` and must return one
    finite numeric relevance score per pair. This keeps the engine independent
    of a particular ML library while allowing a real cross-encoder in production.
    """

    def __init__(self, score_pairs: Callable[[list[tuple[str, str]]], Sequence[float]]) -> None:
        self.score_pairs = score_pairs

    def rerank(self, query: str, documents: Sequence[object], *, top_k: int = 5) -> list[RerankedDocument]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not documents:
            return []
        pairs = [(query, _document_text(doc)) for doc in documents]
        scores = list(self.score_pairs(pairs))
        if len(scores) != len(documents):
            raise ValueError("cross-encoder returned a score count different from the candidate count")
        normalized: list[tuple[float, int, object]] = []
        for index, score in enumerate(scores):
            value = float(score)
            if not math.isfinite(value):
                raise ValueError("cross-encoder returned a non-finite score")
            normalized.append((value, index, documents[index]))
        normalized.sort(key=lambda item: (-item[0], item[1]))
        return [
            RerankedDocument(document=doc, score=score, rank=rank)
            for rank, (score, _, doc) in enumerate(normalized[:top_k], 1)
        ]


class SentenceTransformersCrossEncoder:
    """Optional real neural cross-encoder adapter.

    The heavy dependency is imported lazily. Install the optional `rerank`
    extra to use models from sentence-transformers without making it mandatory
    for CPU-only/basic RAG deployments.
    """

    def __init__(self, model_name: str, *, batch_size: int = 16, device: str | None = None) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RuntimeError(
                "SentenceTransformersCrossEncoder requires the optional rerank dependency"
            ) from error
        kwargs = {"device": device} if device else {}
        self.model = CrossEncoder(model_name, **kwargs)
        self.batch_size = batch_size
        self.model_name = model_name

    def rerank(self, query: str, documents: Sequence[object], *, top_k: int = 5) -> list[RerankedDocument]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not documents:
            return []
        pairs = [(query, _document_text(doc)) for doc in documents]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return CallableCrossEncoderReranker(lambda _: scores).rerank(query, documents, top_k=top_k)


def reciprocal_rank(results: Sequence[RerankedDocument], relevant: set[str]) -> float:
    for result in results:
        citation = result.citation
        if citation in relevant:
            return 1.0 / result.rank
    return 0.0


def ndcg_at_k(results: Sequence[RerankedDocument], relevant: set[str], k: int) -> float:
    if k < 1 or not relevant:
        return 0.0
    dcg = 0.0
    for result in results[:k]:
        if result.citation in relevant:
            dcg += 1.0 / math.log2(result.rank + 1)
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0
