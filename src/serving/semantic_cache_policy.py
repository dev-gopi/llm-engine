"""Production policy primitives for semantic response caching.

The module is backend-neutral.  It supplies privacy/freshness policy, quotas,
negative-cache records, explainable selection metadata and route-specific
threshold resolution without changing the existing local SQLite cache API.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

_SECRET_PATTERNS = (
    re.compile(r"\b(?:sk|rk|pk)_[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(
        r"\b(?:password|passwd|secret|api[_ -]?key|access[_ -]?token)\s*[:=]\s*\S+",
        re.I,
    ),
)
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d .()/-]{8,}\d)(?!\d)")


@dataclass(frozen=True)
class CachePrivacyPolicy:
    no_store: bool = False
    exclude_sensitive: bool = True
    custom_sensitive_patterns: tuple[str, ...] = ()

    def should_bypass(self, prompt: str) -> tuple[bool, str | None]:
        if self.no_store:
            return True, "no_store"
        if not self.exclude_sensitive:
            return False, None
        for pattern in _SECRET_PATTERNS:
            if pattern.search(prompt):
                return True, "sensitive_secret_pattern"
        if _EMAIL.search(prompt):
            return True, "sensitive_email"
        if _PHONE.search(prompt):
            return True, "sensitive_phone"
        for expression in self.custom_sensitive_patterns:
            try:
                if re.search(expression, prompt):
                    return True, "sensitive_custom_pattern"
            except re.error as exc:
                raise ValueError(
                    f"invalid sensitive-data pattern: {expression!r}"
                ) from exc
        return False, None


@dataclass(frozen=True)
class CacheFreshness:
    """Caller-provided fingerprint for dynamic dependencies."""

    value: str = ""
    sources: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "CacheFreshness":
        canonical = "|".join(f"{key}={values[key]}" for key in sorted(values))
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        return cls(value=digest, sources=tuple(sorted(str(k) for k in values)))


@dataclass(frozen=True)
class CacheQuota:
    max_entries: int = 0
    max_bytes: int = 0

    def __post_init__(self) -> None:
        if self.max_entries < 0 or self.max_bytes < 0:
            raise ValueError("cache quotas cannot be negative")


@dataclass(frozen=True)
class ThresholdPolicy:
    default: float = 0.985
    routes: Mapping[str, float] = field(default_factory=dict)
    models: Mapping[str, float] = field(default_factory=dict)
    tasks: Mapping[str, float] = field(default_factory=dict)

    def resolve(
        self,
        *,
        route: str | None = None,
        model: str | None = None,
        task: str | None = None,
    ) -> float:
        value = self.default
        if route and route in self.routes:
            value = self.routes[route]
        elif model and model in self.models:
            value = self.models[model]
        elif task and task in self.tasks:
            value = self.tasks[task]
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("semantic cache threshold must be between 0 and 1")
        return float(value)


@dataclass(frozen=True)
class NegativeCacheEntry:
    key: str
    error_type: str
    message_digest: str
    created_at: float
    expires_at: float

    @classmethod
    def create(
        cls, key: str, error_type: str, message: str, ttl_seconds: float
    ) -> "NegativeCacheEntry":
        now = time.time()
        return cls(
            key,
            error_type,
            hashlib.sha256(message.encode()).hexdigest(),
            now,
            now + ttl_seconds,
        )

    def active(self, now: float | None = None) -> bool:
        return (time.time() if now is None else now) < self.expires_at


@dataclass(frozen=True)
class CacheDebugInfo:
    hit: bool
    kind: str
    similarity: float | None
    threshold: float
    namespace: str
    freshness_fingerprint: str | None = None
    bypass_reason: str | None = None
    candidate_count: int = 0

    def public(self) -> dict[str, Any]:
        return {
            "hit": self.hit,
            "kind": self.kind,
            "similarity": self.similarity,
            "threshold": self.threshold,
            "namespace": self.namespace,
            "freshness_fingerprint": self.freshness_fingerprint,
            "bypass_reason": self.bypass_reason,
            "candidate_count": self.candidate_count,
        }
