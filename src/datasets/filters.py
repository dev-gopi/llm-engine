"""Deterministic, auditable corpus filters for large-scale preparation."""

from __future__ import annotations

import hashlib
import ipaddress
import re
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass


_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()-]{7,}\d)(?!\w)")
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_IPV6 = re.compile(r"(?<![\w:])(?:[0-9A-F]{0,4}:){2,7}[0-9A-F]{0,4}(?![\w:])", re.IGNORECASE)
_US_SSN = re.compile(r"(?<!\d)(?!000|666|9\d\d)\d{3}[- ](?!00)\d{2}[- ](?!0000)\d{4}(?!\d)")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}\b", re.IGNORECASE)
_CREDENTIAL = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16})\b"
    r"|\b(?:api[_ -]?key|access[_ -]?token|client[_ -]?secret)\s*[:=]\s*[\"']?[A-Za-z0-9_./+=-]{12,}[\"']?",
    re.IGNORECASE,
)
_ADDRESS = re.compile(
    r"\b\d{1,6}\s+(?:[A-Z0-9.'-]+\s+){1,6}"
    r"(?:Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Parkway|Pkwy)\b"
    r"(?:[.,]?\s*(?:Apt|Suite|Unit)\s*#?[A-Z0-9-]+)?",
    re.IGNORECASE,
)


@dataclass
class FilterStats:
    accepted: int = 0
    empty: int = 0
    too_short: int = 0
    too_long: int = 0
    low_quality: int = 0
    language: int = 0
    duplicate: int = 0
    near_duplicate: int = 0
    contamination: int = 0
    pii_redactions: int = 0
    email_redactions: int = 0
    phone_redactions: int = 0
    ip_redactions: int = 0
    address_redactions: int = 0
    government_id_redactions: int = 0
    financial_id_redactions: int = 0
    credential_redactions: int = 0


class CorpusFilter:
    def __init__(
        self,
        *,
        min_chars: int = 40,
        max_chars: int = 100_000,
        english_only: bool = False,
        redact_pii: bool = True,
        near_duplicate_distance: int | None = 3,
        excluded_texts: Iterable[str] = (),
        contamination_distance: int | None = 8,
        max_fingerprints: int = 1_000_000,
    ) -> None:
        if max_fingerprints < 1:
            raise ValueError("max_fingerprints must be positive")
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.english_only = english_only
        self.redact_pii = redact_pii
        self.max_fingerprints = max_fingerprints
        self.seen: set[bytes] = set()
        self._seen_order: deque[bytes] = deque()
        self.near_duplicates = _SimilarityIndex(near_duplicate_distance, max_fingerprints)
        self.contamination = _SimilarityIndex(contamination_distance, max_fingerprints)
        self.excluded_digests: set[bytes] = set()
        self.excluded_normalized: list[str] = []
        self.excluded_word_lengths: set[int] = set()
        for excluded in excluded_texts:
            normalized = _normalize(excluded)
            if not normalized:
                continue
            self.excluded_normalized.append(normalized.casefold())
            word_length = len(_words(normalized))
            if word_length >= 5:
                self.excluded_word_lengths.add(word_length)
            self.excluded_digests.add(_digest(normalized))
            fingerprint = _simhash(normalized)
            if fingerprint is not None:
                self.contamination.add(fingerprint)
        self.stats = FilterStats()

    def apply(self, text: str) -> str | None:
        text = _normalize(text)
        if not text:
            self.stats.empty += 1
            return None
        if len(text) < self.min_chars:
            self.stats.too_short += 1
            return None
        if len(text) > self.max_chars:
            self.stats.too_long += 1
            return None
        printable = sum(character.isprintable() for character in text) / len(text)
        alphanumeric = sum(character.isalnum() for character in text) / len(text)
        if printable < 0.95 or alphanumeric < 0.25:
            self.stats.low_quality += 1
            return None
        if self.english_only:
            letters = [character for character in text if character.isalpha()]
            ascii_ratio = sum(character.isascii() for character in letters) / max(len(letters), 1)
            if ascii_ratio < 0.85:
                self.stats.language += 1
                return None
        digest = _digest(text)
        fingerprint = _simhash(text)
        if digest in self.excluded_digests or self._is_contaminated(text, fingerprint):
            self.stats.contamination += 1
            return None
        if digest in self.seen:
            self.stats.duplicate += 1
            return None
        if fingerprint is not None and self.near_duplicates.contains(fingerprint):
            self.stats.near_duplicate += 1
            return None
        self._remember_digest(digest)
        if fingerprint is not None:
            self.near_duplicates.add(fingerprint)
        if self.redact_pii:
            text = self._redact_pii(text)
        self.stats.accepted += 1
        return text

    def _remember_digest(self, digest: bytes) -> None:
        self.seen.add(digest)
        self._seen_order.append(digest)
        if len(self._seen_order) > self.max_fingerprints:
            self.seen.discard(self._seen_order.popleft())

    def _is_contaminated(self, text: str, fingerprint: int | None) -> bool:
        folded = text.casefold()
        if any(excluded in folded for excluded in self.excluded_normalized):
            return True
        if fingerprint is not None and self.contamination.contains(fingerprint):
            return True
        words = _words(text)
        for width in self.excluded_word_lengths:
            if len(words) <= width:
                continue
            stride = max(1, width // 4)
            starts = list(range(0, len(words) - width + 1, stride))
            final_start = len(words) - width
            if not starts or starts[-1] != final_start:
                starts.append(final_start)
            for start in starts:
                window = " ".join(words[start:start + width])
                window_hash = _simhash(window)
                if window_hash is not None and self.contamination.contains(window_hash):
                    return True
        return False

    def _redact_pii(self, text: str) -> str:
        text, credentials = _CREDENTIAL.subn("<credential>", text)
        text, government_ids = _US_SSN.subn("<government-id>", text)
        text, cards = _validated_sub(_CARD, text, "<financial-id>", _luhn_valid)
        text, ibans = _validated_sub(_IBAN, text, "<financial-id>", _iban_valid)
        text, emails = _EMAIL.subn("<email>", text)
        text, ipv4 = _validated_sub(_IPV4, text, "<ip>", _ip_valid)
        text, ipv6 = _validated_sub(_IPV6, text, "<ip>", _ip_valid)
        text, phones = _validated_sub(_PHONE, text, "<phone>", _phone_valid)
        text, addresses = _ADDRESS.subn("<address>", text)
        self.stats.credential_redactions += credentials
        self.stats.government_id_redactions += government_ids
        self.stats.financial_id_redactions += cards + ibans
        self.stats.email_redactions += emails
        self.stats.ip_redactions += ipv4 + ipv6
        self.stats.phone_redactions += phones
        self.stats.address_redactions += addresses
        self.stats.pii_redactions += (
            credentials + government_ids + cards + ibans + emails + ipv4 + ipv6
            + phones + addresses
        )
        return text


class _SimilarityIndex:
    """Bounded SimHash LSH index with deterministic FIFO eviction."""

    def __init__(self, distance: int | None, capacity: int) -> None:
        if distance is not None and not 0 <= distance <= 15:
            raise ValueError("similarity distance must be between 0 and 15, or None")
        self.distance = distance
        self.capacity = capacity
        self.band_count = (distance + 1) if distance is not None else 0
        self.buckets: dict[tuple[int, int], set[int]] = {}
        self.order: deque[int] = deque()

    def contains(self, fingerprint: int) -> bool:
        if self.distance is None:
            return False
        candidates: set[int] = set()
        for key in self._keys(fingerprint):
            candidates.update(self.buckets.get(key, ()))
        return any((candidate ^ fingerprint).bit_count() <= self.distance for candidate in candidates)

    def add(self, fingerprint: int) -> None:
        if self.distance is None:
            return
        self.order.append(fingerprint)
        for key in self._keys(fingerprint):
            self.buckets.setdefault(key, set()).add(fingerprint)
        if len(self.order) > self.capacity:
            expired = self.order.popleft()
            for key in self._keys(expired):
                bucket = self.buckets[key]
                bucket.discard(expired)
                if not bucket:
                    del self.buckets[key]

    def _keys(self, fingerprint: int):
        for band in range(self.band_count):
            start = band * 64 // self.band_count
            end = (band + 1) * 64 // self.band_count
            mask = (1 << (end - start)) - 1
            yield band, (fingerprint >> start) & mask


def _normalize(text: str) -> str:
    return " ".join(str(text).split())


def _digest(text: str) -> bytes:
    return hashlib.blake2b(text.casefold().encode("utf-8"), digest_size=16).digest()


def _simhash(text: str) -> int | None:
    words = _words(text)
    if len(words) < 5:
        return None
    features = [" ".join(words[index:index + 2]) for index in range(len(words) - 1)]
    weights = [0] * 64
    for feature in features:
        value = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest())
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    fingerprint = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            fingerprint |= 1 << bit
    return fingerprint


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def _validated_sub(pattern: re.Pattern, text: str, replacement: str, validator) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match) -> str:
        nonlocal count
        if validator(match.group(0)):
            count += 1
            return replacement
        return match.group(0)

    return pattern.sub(replace, text), count


def _luhn_valid(value: str) -> bool:
    digits = [int(character) for character in value if character.isdigit()]
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _iban_valid(value: str) -> bool:
    compact = re.sub(r"\s+", "", value).upper()
    if not 15 <= len(compact) <= 34:
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(ord(character) - 55) if character.isalpha() else character for character in rearranged)
    return int(numeric) % 97 == 1


def _ip_valid(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


def _phone_valid(value: str) -> bool:
    if value.count(".") == 3:
        return False
    digit_count = sum(character.isdigit() for character in value)
    return 8 <= digit_count <= 15
