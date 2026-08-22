"""Core byte-level Byte Pair Encoding operations."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache


def bytes_to_unicode() -> dict[int, str]:
    """Return a reversible mapping that avoids control characters in artifacts."""

    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("¡"), ord("¬") + 1))
    visible += list(range(ord("®"), ord("ÿ") + 1))
    byte_values = visible[:]
    code_points = visible[:]
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            code_points.append(256 + extra)
            extra += 1
    return dict(zip(byte_values, map(chr, code_points), strict=True))


BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {character: value for value, character in BYTE_ENCODER.items()}


def adjacent_pairs(symbols: tuple[str, ...]) -> set[tuple[str, str]]:
    return set(zip(symbols, symbols[1:]))


def merge_pair(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    """Merge every non-overlapping occurrence of ``pair`` in ``symbols``."""

    merged: list[str] = []
    index = 0
    while index < len(symbols):
        if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == pair:
            merged.append(symbols[index] + symbols[index + 1])
            index += 2
        else:
            merged.append(symbols[index])
            index += 1
    return tuple(merged)


class BPE:
    """Apply an ordered list of learned BPE merges to byte-level symbols."""

    def __init__(self, merges: Iterable[tuple[str, str]], cache_size: int = 65_536):
        self.merges = tuple(merges)
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._apply_cached = lru_cache(maxsize=cache_size)(self._apply_uncached)

    def apply(self, symbols: tuple[str, ...]) -> tuple[str, ...]:
        if not symbols:
            return ()
        return self._apply_cached(symbols)

    def _apply_uncached(self, symbols: tuple[str, ...]) -> tuple[str, ...]:
        while len(symbols) > 1:
            pair = min(
                adjacent_pairs(symbols),
                key=lambda item: self.ranks.get(item, float("inf")),
            )
            if pair not in self.ranks:
                break
            symbols = merge_pair(symbols, pair)
        return symbols

    def clear_cache(self) -> None:
        self._apply_cached.cache_clear()
