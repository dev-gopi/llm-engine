"""Memory-conscious byte-level BPE vocabulary training."""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import regex

from .bpe import BYTE_ENCODER, merge_pair
from .encoder import DEFAULT_PATTERN, DEFAULT_SPECIAL_TOKENS, Tokenizer


Pair = tuple[str, str]
ProgressCallback = Callable[[int, int, Pair, int], None]


@dataclass(frozen=True)
class TrainingStats:
    documents: int
    utf8_bytes: int
    pre_tokens: int
    unique_pre_tokens: int
    merges: int


class BPETokenizerTrainer:
    """Train byte-level BPE using incrementally maintained pair frequencies.

    Pair counts are updated only for word types touched by each merge. This
    avoids rescanning the complete vocabulary on every iteration and makes the
    implementation suitable for substantially larger corpora than a naive BPE
    trainer, while remaining deterministic and dependency-light.
    """

    def __init__(
        self,
        vocab_size: int = 50_000,
        *,
        min_frequency: int = 2,
        special_tokens: Iterable[str] = DEFAULT_SPECIAL_TOKENS,
        pattern: str = DEFAULT_PATTERN,
        max_training_bytes: int | None = None,
    ):
        self.special_token_names = tuple(dict.fromkeys(special_tokens))
        minimum_size = len(self.special_token_names) + 256
        if vocab_size < minimum_size:
            raise ValueError(f"vocab_size must be at least {minimum_size} for byte-level BPE")
        if min_frequency < 1:
            raise ValueError("min_frequency must be positive")
        if max_training_bytes is not None and max_training_bytes < 1:
            raise ValueError("max_training_bytes must be positive when provided")

        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.pattern = pattern
        self._pattern = regex.compile(pattern)
        self.max_training_bytes = max_training_bytes
        self.stats: TrainingStats | None = None

    def train(
        self,
        texts: Iterable[str],
        *,
        progress: ProgressCallback | None = None,
    ) -> Tokenizer:
        frequencies, documents, utf8_bytes = self._count_pre_tokens(texts)
        if not frequencies:
            raise ValueError("training corpus contains no text tokens")

        words = [
            tuple(BYTE_ENCODER[value] for value in token.encode("utf-8"))
            for token in frequencies
        ]
        word_frequencies = list(frequencies.values())
        pair_counts: Counter[Pair] = Counter()
        pair_to_words: dict[Pair, set[int]] = defaultdict(set)

        for word_id, (symbols, frequency) in enumerate(zip(words, word_frequencies, strict=True)):
            occurrences = Counter(zip(symbols, symbols[1:]))
            for pair, count in occurrences.items():
                pair_counts[pair] += count * frequency
                pair_to_words[pair].add(word_id)

        heap = [(-count, pair[0], pair[1]) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        special_tokens = {token: index for index, token in enumerate(self.special_token_names)}
        vocab = dict(special_tokens)
        for value in range(256):
            vocab[BYTE_ENCODER[value]] = len(vocab)

        merges: list[Pair] = []
        target_merges = self.vocab_size - len(vocab)
        while len(vocab) < self.vocab_size:
            selected = self._pop_best_pair(heap, pair_counts)
            if selected is None:
                break
            pair, frequency = selected
            if frequency < self.min_frequency:
                break

            affected_words = tuple(pair_to_words.get(pair, ()))
            if not affected_words:
                pair_counts.pop(pair, None)
                continue

            for word_id in affected_words:
                old_symbols = words[word_id]
                new_symbols = merge_pair(old_symbols, pair)
                if new_symbols == old_symbols:
                    continue
                words[word_id] = new_symbols
                self._update_pair_statistics(
                    word_id,
                    old_symbols,
                    new_symbols,
                    word_frequencies[word_id],
                    pair_counts,
                    pair_to_words,
                    heap,
                )

            merges.append(pair)
            merged_symbol = pair[0] + pair[1]
            if merged_symbol not in vocab:
                vocab[merged_symbol] = len(vocab)
            if progress and (len(merges) == 1 or len(merges) % 100 == 0):
                progress(len(merges), target_merges, pair, frequency)

        pre_token_count = sum(frequencies.values())
        self.stats = TrainingStats(
            documents=documents,
            utf8_bytes=utf8_bytes,
            pre_tokens=pre_token_count,
            unique_pre_tokens=len(frequencies),
            merges=len(merges),
        )
        metadata = {
            "trainer": "incremental_byte_level_bpe",
            "requested_vocab_size": self.vocab_size,
            "actual_vocab_size": len(vocab),
            "min_frequency": self.min_frequency,
            "training_stats": {
                "documents": documents,
                "utf8_bytes": utf8_bytes,
                "pre_tokens": pre_token_count,
                "unique_pre_tokens": len(frequencies),
                "merges": len(merges),
            },
        }
        return Tokenizer(
            vocab,
            merges,
            special_tokens=special_tokens,
            pattern=self.pattern,
            metadata=metadata,
        )

    def _count_pre_tokens(self, texts: Iterable[str]) -> tuple[Counter[str], int, int]:
        frequencies: Counter[str] = Counter()
        documents = 0
        utf8_bytes = 0
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("training documents must be strings")
            encoded_size = len(text.encode("utf-8"))
            if self.max_training_bytes is not None and utf8_bytes + encoded_size > self.max_training_bytes:
                break
            frequencies.update(match.group(0) for match in self._pattern.finditer(text))
            documents += 1
            utf8_bytes += encoded_size
        return frequencies, documents, utf8_bytes

    @staticmethod
    def _pop_best_pair(
        heap: list[tuple[int, str, str]], pair_counts: Counter[Pair]
    ) -> tuple[Pair, int] | None:
        while heap:
            negative_count, left, right = heapq.heappop(heap)
            pair = (left, right)
            current_count = pair_counts.get(pair, 0)
            if current_count == -negative_count and current_count > 0:
                return pair, current_count
        return None

    @staticmethod
    def _update_pair_statistics(
        word_id: int,
        old_symbols: tuple[str, ...],
        new_symbols: tuple[str, ...],
        word_frequency: int,
        pair_counts: Counter[Pair],
        pair_to_words: dict[Pair, set[int]],
        heap: list[tuple[int, str, str]],
    ) -> None:
        old_occurrences = Counter(zip(old_symbols, old_symbols[1:]))
        new_occurrences = Counter(zip(new_symbols, new_symbols[1:]))
        changed_pairs = old_occurrences.keys() | new_occurrences.keys()

        for candidate in changed_pairs:
            delta = (new_occurrences[candidate] - old_occurrences[candidate]) * word_frequency
            if delta:
                pair_counts[candidate] += delta
            if new_occurrences[candidate]:
                pair_to_words[candidate].add(word_id)
            else:
                pair_to_words[candidate].discard(word_id)
                if not pair_to_words[candidate]:
                    pair_to_words.pop(candidate, None)
            current_count = pair_counts.get(candidate, 0)
            if current_count > 0:
                heapq.heappush(heap, (-current_count, candidate[0], candidate[1]))
            else:
                pair_counts.pop(candidate, None)
