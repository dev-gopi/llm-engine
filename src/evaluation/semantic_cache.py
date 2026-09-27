"""Offline semantic-cache false-positive and answer-drift evaluation."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Iterable


@dataclass(frozen=True)
class SemanticCacheEvalCase:
    query: str
    cached_query: str
    expected_same_answer: bool
    answer_match: bool
    similarity: float


@dataclass(frozen=True)
class SemanticCacheEvalReport:
    cases: int
    false_positive_rate: float
    answer_drift_rate: float
    mean_similarity: float
    threshold: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_semantic_cache(
    cases: Iterable[SemanticCacheEvalCase],
    *,
    threshold: float,
) -> SemanticCacheEvalReport:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    rows = list(cases)
    if not rows:
        return SemanticCacheEvalReport(0, 0.0, 0.0, 0.0, threshold)
    accepted = [row for row in rows if row.similarity >= threshold]
    false_positive = [row for row in accepted if not row.expected_same_answer]
    drift = [row for row in accepted if row.expected_same_answer and not row.answer_match]
    return SemanticCacheEvalReport(
        cases=len(rows),
        false_positive_rate=len(false_positive) / len(accepted) if accepted else 0.0,
        answer_drift_rate=len(drift) / len(accepted) if accepted else 0.0,
        mean_similarity=sum(row.similarity for row in rows) / len(rows),
        threshold=threshold,
    )


def evaluate_from_pairs(
    rows: Iterable[tuple[str, str, bool, bool]],
    similarity: Callable[[str, str], float],
    *,
    threshold: float,
) -> SemanticCacheEvalReport:
    cases = [
        SemanticCacheEvalCase(query, cached, expected, answer_match, float(similarity(query, cached)))
        for query, cached, expected, answer_match in rows
    ]
    return evaluate_semantic_cache(cases, threshold=threshold)
