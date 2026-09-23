"""Independent train/evaluation contamination and near-duplicate auditing."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from .loader import iter_records
from .preprocessor import record_to_text

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def normalize_for_audit(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(_WORD_RE.findall(text))


def exact_fingerprint(text: str) -> str:
    return hashlib.sha256(normalize_for_audit(text).encode("utf-8")).hexdigest()


def shingles(text: str, size: int = 5) -> set[str]:
    tokens = normalize_for_audit(text).split()
    if size < 1:
        raise ValueError("shingle size must be positive")
    if len(tokens) <= size:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + size]) for i in range(len(tokens) - size + 1)}


def _hash_shingle(value: str, seed: int) -> int:
    return int.from_bytes(hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=8).digest(), "big")


def minhash_signature(values: Iterable[str], *, num_hashes: int = 32) -> tuple[int, ...]:
    if num_hashes < 4:
        raise ValueError("num_hashes must be at least 4")
    values = list(values)
    if not values:
        return (2**64 - 1,) * num_hashes
    return tuple(min(_hash_shingle(value, seed) for value in values) for seed in range(num_hashes))


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


@dataclass(frozen=True)
class AuditDocument:
    split: str
    path: str
    line: int
    text: str
    fingerprint: str
    shingles: frozenset[str]
    signature: tuple[int, ...]


def load_documents(paths: Iterable[str], split: str, max_records: int, shingle_size: int) -> list[AuditDocument]:
    if max_records < 1:
        raise ValueError("max_records must be positive")
    result: list[AuditDocument] = []
    for path in paths:
        for line, record in enumerate(iter_records(path), 1):
            if len(result) >= max_records:
                break
            text = record if isinstance(record, str) else record_to_text(record)
            normalized = normalize_for_audit(text)
            if not normalized:
                continue
            values = shingles(text, shingle_size)
            result.append(AuditDocument(split, str(path), line, text, exact_fingerprint(text),
                                        frozenset(values), minhash_signature(values)))
        if len(result) >= max_records:
            break
    return result


def _candidate_pairs(train: list[AuditDocument], evaluation: list[AuditDocument]) -> set[tuple[int, int]]:
    """Use deterministic MinHash banding to avoid quadratic all-pairs comparison."""
    bands: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    rows = 4
    for index, document in enumerate(train):
        for start in range(0, len(document.signature), rows):
            key = (start, document.signature[start:start + rows])
            bands.setdefault(key, []).append(index)
    pairs: set[tuple[int, int]] = set()
    for eval_index, document in enumerate(evaluation):
        for start in range(0, len(document.signature), rows):
            key = (start, document.signature[start:start + rows])
            for train_index in bands.get(key, ()):
                pairs.add((train_index, eval_index))
    return pairs


def audit_contamination(train: list[AuditDocument], evaluation: list[AuditDocument], *, threshold: float = 0.85) -> dict:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    train_by_hash: dict[str, list[int]] = {}
    for index, item in enumerate(train):
        train_by_hash.setdefault(item.fingerprint, []).append(index)
    exact_matches = []
    for eval_index, item in enumerate(evaluation):
        for train_index in train_by_hash.get(item.fingerprint, ()):
            exact_matches.append({"train": train[train_index].path + f":{train[train_index].line}",
                                  "evaluation": item.path + f":{item.line}"})
    near_matches = []
    for train_index, eval_index in sorted(_candidate_pairs(train, evaluation)):
        score = jaccard(set(train[train_index].shingles), set(evaluation[eval_index].shingles))
        if score >= threshold and train[train_index].fingerprint != evaluation[eval_index].fingerprint:
            near_matches.append({"train": train[train_index].path + f":{train[train_index].line}",
                                  "evaluation": evaluation[eval_index].path + f":{evaluation[eval_index].line}",
                                  "jaccard": round(score, 6)})
    return {
        "train_records": len(train),
        "evaluation_records": len(evaluation),
        "exact_matches": exact_matches,
        "near_duplicate_matches": near_matches,
        "exact_match_count": len(exact_matches),
        "near_duplicate_count": len(near_matches),
        "near_duplicate_threshold": threshold,
        "status": "failed" if exact_matches or near_matches else "passed",
    }
