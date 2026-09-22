"""Candidate retrieval, reranking, and ranking evaluation."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .reranker import (
    LexicalCrossEncoderBaseline,
    RerankedDocument,
    Reranker,
    ndcg_at_k,
    reciprocal_rank,
)


@dataclass(frozen=True)
class RetrievalEvaluation:
    candidate_count: int
    reranked_count: int
    recall_at_k: float
    reranked_recall_at_k: float
    mrr: float
    ndcg_at_k: float

    @property
    def retrieved(self) -> int:
        return self.candidate_count

    @property
    def reranked(self) -> int:
        return self.reranked_count


class RetrieverPipeline:
    """Retrieve a broad candidate pool, rerank it, then expose only top-K."""

    def __init__(self, retriever, reranker: Reranker | None = None):
        self.retriever = retriever
        self.reranker = reranker or LexicalCrossEncoderBaseline()

    def search(self, query: str, *, candidate_k: int = 20, top_k: int = 5) -> list[RerankedDocument]:
        if candidate_k < top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        candidates = self.retriever.search(query, top_k=candidate_k)
        return self.reranker.rerank(query, candidates, top_k=top_k)

    @staticmethod
    def evaluate(
        candidates: Sequence[object],
        reranked: Sequence[RerankedDocument],
        relevant_sources: set[str],
        *,
        k: int,
    ) -> RetrievalEvaluation:
        if k < 1:
            raise ValueError("k must be positive")
        if not relevant_sources:
            return RetrievalEvaluation(len(candidates), len(reranked), 0.0, 0.0, 0.0, 0.0)
        candidate_sources = {
            getattr(item, "url", None) or getattr(item, "source", None)
            for item in candidates
        }
        recall = len(candidate_sources & relevant_sources) / len(relevant_sources)
        reranked_sources = {
            getattr(getattr(item, "document", item), "url", None)
            or getattr(getattr(item, "document", item), "source", None)
            for item in reranked[:k]
        }
        reranked_recall = len(reranked_sources & relevant_sources) / len(relevant_sources)
        return RetrievalEvaluation(
            candidate_count=len(candidates),
            reranked_count=len(reranked),
            recall_at_k=recall,
            reranked_recall_at_k=reranked_recall,
            mrr=reciprocal_rank(reranked[:k], relevant_sources),
            ndcg_at_k=ndcg_at_k(reranked, relevant_sources, k),
        )

    @staticmethod
    def recall_at_k(results: Sequence[object], relevant_sources: set[str], k: int) -> float:
        if k < 1:
            raise ValueError("k must be positive")
        if not relevant_sources:
            return 0.0
        seen = {
            getattr(getattr(result, "document", result), "url", None)
            or getattr(getattr(result, "document", result), "source", None)
            for result in results[:k]
        }
        return len(seen & relevant_sources) / len(relevant_sources)
