"""Deterministic per-document quality scoring and source weighting."""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping

from .loader import iter_records
from .preprocessor import record_to_text

_WORD_RE = re.compile(r"\S+", re.UNICODE)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _repetition_ratio(tokens: list[str]) -> float:
    if len(tokens) < 4:
        return 0.0
    trigrams = [tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)]
    counts = Counter(trigrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(trigrams)


def score_text(text: str) -> dict[str, float | int]:
    normalized = unicodedata.normalize("NFKC", text).strip()
    tokens = _WORD_RE.findall(normalized)
    chars = len(normalized)
    letters = sum(character.isalpha() for character in normalized)
    controls = sum(unicodedata.category(character).startswith("C") for character in normalized)
    unique_ratio = len(set(token.casefold() for token in tokens)) / len(tokens) if tokens else 0.0
    repetition = _repetition_ratio([token.casefold() for token in tokens])
    alpha_ratio = letters / chars if chars else 0.0
    control_ratio = controls / chars if chars else 0.0
    length_score = 1.0 if 16 <= len(tokens) <= 4096 else (0.5 if tokens else 0.0)
    score = (
        0.20 * length_score
        + 0.20 * _clamp(unique_ratio)
        + 0.20 * _clamp(1.0 - repetition)
        + 0.20 * _clamp(alpha_ratio)
        + 0.20 * _clamp(1.0 - control_ratio * 10.0)
    )
    return {
        "score": round(_clamp(score), 6),
        "tokens": len(tokens),
        "characters": chars,
        "unique_token_ratio": round(unique_ratio, 6),
        "repetition_ratio": round(repetition, 6),
        "alphabetic_ratio": round(alpha_ratio, 6),
        "control_ratio": round(control_ratio, 6),
    }


def score_records(path: str, *, max_records: int | None = None) -> tuple[list[dict], dict[str, float]]:
    rows: list[dict] = []
    for line, record in enumerate(iter_records(path), 1):
        if max_records is not None and len(rows) >= max_records:
            break
        text = record if isinstance(record, str) else record_to_text(record)
        metrics = score_text(text)
        document_id = str(record.get("id", f"{path}:{line}")) if isinstance(record, Mapping) else f"{path}:{line}"
        rows.append({"document_id": document_id, "path": str(path), "line": line, **metrics})
    mean = sum(row["score"] for row in rows) / len(rows) if rows else 0.0
    return rows, {"documents": len(rows), "mean_quality": round(mean, 6)}


def quality_source_weights(source_quality: Mapping[str, float], priors: Mapping[str, float] | None = None) -> dict[str, float]:
    if not source_quality:
        raise ValueError("source_quality must not be empty")
    priors = priors or {name: 1.0 for name in source_quality}
    raw: dict[str, float] = {}
    for name, quality in source_quality.items():
        prior = float(priors.get(name, 0.0))
        quality = float(quality)
        if not math.isfinite(quality) or not 0 <= quality <= 1 or not math.isfinite(prior) or prior < 0:
            raise ValueError("quality scores must be in [0,1] and priors must be non-negative")
        raw[name] = quality * prior
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("quality-weighted source weights must have positive total")
    return {name: round(value / total, 12) for name, value in raw.items()}
