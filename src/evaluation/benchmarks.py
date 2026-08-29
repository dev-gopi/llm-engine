"""Simple reproducible scoring for held-out generation benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import regex


@dataclass(frozen=True)
class BenchmarkCase:
    category: str
    prompt: str
    expected: tuple[str, ...]
    forbidden: tuple[str, ...] = ()


def normalize_answer(text: str) -> str:
    return " ".join(regex.findall(r"[\p{L}\p{M}\p{N}]+", text.casefold()))


def score_answer(answer: str, case: BenchmarkCase) -> float:
    normalized = normalize_answer(answer)
    expected = [normalize_answer(value) for value in case.expected]
    forbidden = [normalize_answer(value) for value in case.forbidden]
    if any(value and value in normalized for value in forbidden):
        return 0.0
    return float(any(value and value in normalized for value in expected))


def summarize_scores(results: list[tuple[BenchmarkCase, float]]) -> dict[str, float | int]:
    categories: dict[str, list[float]] = {}
    for case, score in results:
        categories.setdefault(case.category, []).append(score)
    summary: dict[str, float | int] = {
        "cases": len(results),
        "accuracy": sum(score for _, score in results) / max(len(results), 1),
    }
    summary.update({
        f"accuracy_{category}": sum(scores) / len(scores)
        for category, scores in sorted(categories.items())
    })
    return summary
