"""Simple reproducible scoring for held-out generation benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import regex
from decimal import Decimal, InvalidOperation


@dataclass(frozen=True)
class BenchmarkCase:
    category: str
    prompt: str
    expected: tuple[str, ...]
    forbidden: tuple[str, ...] = ()
    match: str = "contains"
    max_answer_tokens: int | None = None
    require_thinking_trace: bool = False

    def __post_init__(self) -> None:
        if self.match not in {"contains", "exact", "exact_code", "number", "final_number"}:
            raise ValueError("benchmark match must be contains, exact, exact_code, number, or final_number")
        if not self.expected or any(not value.strip() for value in self.expected):
            raise ValueError("benchmark expected answers must be nonempty")
        if self.max_answer_tokens is not None and self.max_answer_tokens < 1:
            raise ValueError("benchmark max_answer_tokens must be positive")


def normalize_answer(text: str) -> str:
    return " ".join(regex.findall(r"[\p{L}\p{M}\p{N}]+", text.casefold()))


def score_answer(answer: str, case: BenchmarkCase) -> float:
    if case.require_thinking_trace:
        opening = answer.find("<thinking>")
        closing = answer.find("</thinking>")
        if opening < 0 or closing <= opening + len("<thinking>"):
            return 0.0
        if answer.find("<thinking>", opening + 1) >= 0 or answer.find("</thinking>", closing + 1) >= 0:
            return 0.0
        answer = answer[closing + len("</thinking>"):].strip()
        if not answer:
            return 0.0
    answer_tokens = normalize_answer(answer).split()
    if case.max_answer_tokens is not None and len(answer_tokens) > case.max_answer_tokens:
        return 0.0
    expected = [normalize_answer(value).split() for value in case.expected]
    forbidden = [normalize_answer(value).split() for value in case.forbidden]
    if any(_contains_tokens(answer_tokens, value) for value in forbidden if value):
        return 0.0
    if case.match == "exact_code":
        # Preserve operators: tokenizing only letters/numbers treated a+b and
        # a%b as equivalent. Only harmless surrounding Markdown is ignored.
        code = answer.strip()
        if code.startswith("```") and code.endswith("```"):
            code = code.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        code = code.strip("`").strip()
        return float(any(code == value.strip() for value in case.expected))
    if case.match in {"number", "final_number"}:
        # These cases explicitly request a number only. Do not reward a wrong
        # solution just because the expected number appears among its steps.
        def number(text):
            value = text.strip().rstrip(".").strip()
            if not regex.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)", value):
                return None
            try:
                return Decimal(value)
            except InvalidOperation:
                return None
        if case.match == "final_number":
            matches = regex.findall(r"####\s*([+-]?[0-9][0-9,]*(?:\.[0-9]+)?)", answer)
            answer = matches[-1].replace(",", "") if matches else ""
        actual = number(answer)
        return float(actual is not None and any(actual == number(value) for value in case.expected))
    if case.match == "exact":
        return float(any(answer_tokens == value for value in expected if value))
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


def compare_reports(baseline: dict, candidate: dict) -> dict:
    """Reject protocol changes and any lost previously passing probe.

    This is a deterministic regression gate, not a statistical significance
    test or evidence that this small probe set covers all capabilities.
    """
    if not baseline.get("protocol") or baseline["protocol"] != candidate.get("protocol"):
        raise ValueError("evaluation protocol differs; rerun both checkpoints with the same cases and settings")
    def indexed(report):
        rows = report["results"]
        values = {(row["category"], row["prompt"]): float(row["score"]) for row in rows}
        if len(values) != len(rows) or not values or any(score not in (0.0, 1.0) for score in values.values()):
            raise ValueError("results require unique cases and binary scores")
        return values
    before, after = indexed(baseline), indexed(candidate)
    if before.keys() != after.keys():
        raise ValueError("evaluation case coverage differs")
    regressions = [dict(category=key[0], prompt=key[1]) for key in before if after[key] < before[key]]
    domains = sorted({key[0] for key in before})
    delta = {domain: sum(after[key] - before[key] for key in before if key[0] == domain)
             / sum(key[0] == domain for key in before) for domain in domains}
    return {"passed": not regressions, "regressions": regressions,
            "accuracy_delta": sum(after[key] - before[key] for key in before) / len(before),
            "domain_accuracy_deltas": delta}
