"""Retrieval, reranking, faithfulness and citation benchmark primitives."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag.reranker import Reranker


@dataclass(frozen=True)
class RAGBenchmarkCase:
    query: str
    relevant_ids: frozenset[str]
    supported_facts: tuple[str, ...] = ()


@dataclass(frozen=True)
class RAGBenchmarkReport:
    queries: int
    retrieval_recall_at_k: float
    reranked_recall_at_k: float
    mrr: float
    ndcg: float
    faithfulness: float
    citation_correctness: float

    @property
    def rerank_improvement(self) -> float:
        return self.reranked_recall_at_k - self.retrieval_recall_at_k

    def to_dict(self) -> dict[str, float | int]:
        return {
            "queries": self.queries,
            "retrieval_recall_at_k": self.retrieval_recall_at_k,
            "reranked_recall_at_k": self.reranked_recall_at_k,
            "rerank_improvement": self.rerank_improvement,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "faithfulness": self.faithfulness,
            "citation_correctness": self.citation_correctness,
        }


def _reciprocal_rank(order: Sequence[str], relevant: frozenset[str]) -> float:
    for index, identifier in enumerate(order, 1):
        if identifier in relevant:
            return 1.0 / index
    return 0.0


def _ndcg(order: Sequence[str], relevant: frozenset[str]) -> float:
    import math
    dcg = sum((1.0 / math.log2(index + 2)) for index, identifier in enumerate(order) if identifier in relevant)
    ideal = sum((1.0 / math.log2(index + 2)) for index in range(min(len(relevant), len(order))))
    return dcg / ideal if ideal else 0.0


def evaluate_rag(
    cases: Sequence[RAGBenchmarkCase],
    candidate_lists: Sequence[Sequence[object]],
    reranker: Reranker,
    *,
    top_k: int = 5,
    answers: Sequence[str] | None = None,
) -> RAGBenchmarkReport:
    if len(cases) != len(candidate_lists):
        raise ValueError("cases and candidate_lists must have equal length")
    if answers is not None and len(answers) != len(cases):
        raise ValueError("answers must match benchmark cases")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    retrieval_hits = 0
    reranked_hits = 0
    reciprocal = 0.0
    ndcg = 0.0
    faithful = 0
    cited = 0
    for index, (case, candidates) in enumerate(zip(cases, candidate_lists, strict=True)):
        initial = [doc.id for doc in candidates[:top_k]]
        retrieval_hits += int(bool(case.relevant_ids.intersection(initial)))
        ranked = reranker.rerank(case.query, candidates, top_k=top_k)
        order = [doc.document.id for doc in ranked]
        reranked_hits += int(bool(case.relevant_ids.intersection(order)))
        reciprocal += _reciprocal_rank(order, case.relevant_ids)
        ndcg += _ndcg(order, case.relevant_ids)
        if answers is not None:
            answer = answers[index].casefold()
            supported = [fact.casefold() for fact in case.supported_facts]
            ok = bool(supported) and any(fact in answer for fact in supported)
            faithful += int(ok)
            cited += int(ok and any(doc_id in answer for doc_id in case.relevant_ids))

    count = max(len(cases), 1)
    answer_count = len(cases) if answers is not None else 0
    return RAGBenchmarkReport(
        queries=len(cases),
        retrieval_recall_at_k=retrieval_hits / count,
        reranked_recall_at_k=reranked_hits / count,
        mrr=reciprocal / count,
        ndcg=ndcg / count,
        faithfulness=faithful / max(answer_count, 1),
        citation_correctness=cited / max(answer_count, 1),
    )
