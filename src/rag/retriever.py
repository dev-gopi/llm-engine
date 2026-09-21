"""Retrieval/reranking pipeline contracts."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from inference.rag import RetrievalResult
from .reranker import Reranker, LexicalCrossEncoderBaseline, RerankedDocument

@dataclass(frozen=True)
class RetrievalEvaluation:
    retrieved: int
    reranked: int
    recall_at_k: float

class RetrieverPipeline:
    def __init__(self, retriever, reranker: Reranker | None = None):
        self.retriever, self.reranker = retriever, reranker or LexicalCrossEncoderBaseline()
    def search(self, query: str, *, candidate_k: int = 20, top_k: int = 5):
        candidates = self.retriever.search(query, top_k=candidate_k)
        return self.reranker.rerank(query, candidates, top_k=top_k)
    @staticmethod
    def recall_at_k(results: Sequence[object], relevant_sources: set[str], k: int) -> float:
        if not relevant_sources: return 0.0
        seen = {getattr(getattr(r, "document", r), "url", None) for r in results[:k]}
        return len(seen & relevant_sources) / len(relevant_sources)
