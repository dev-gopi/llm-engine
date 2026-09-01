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
    answer_tokens = normalize_answer(answer).split()
    expected = [normalize_answer(value).split() for value in case.expected]
    forbidden = [normalize_answer(value).split() for value in case.forbidden]
    if any(_contains_tokens(answer_tokens, value) for value in forbidden if value):
        return 0.0
    return float(any(_contains_tokens(answer_tokens, value) for value in expected if value))


def _contains_tokens(tokens: list[str], phrase: list[str]) -> bool:
    """Return whether a normalized token phrase occurs contiguously."""
    width = len(phrase)
    return width > 0 and any(
        tokens[index : index + width] == phrase
        for index in range(len(tokens) - width + 1)
    )


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
