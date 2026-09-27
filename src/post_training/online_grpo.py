"""Online GRPO rollout generation and bounded replay utilities."""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from inference.generator import Generator


class CompletionScorer(Protocol):
    def score(
        self, prompt: str, completion: str, *, normalized: bool = False
    ) -> float: ...


@dataclass(frozen=True)
class RolloutConfig:
    group_size: int = 4
    max_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 3
    min_tokens: int = 1
    reward_model_weight: float = 1.0
    exact_match_weight: float = 0.0
    normalize_reward_model: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        if self.group_size < 2:
            raise ValueError("online GRPO group_size must be at least 2")
        if self.max_tokens < 1:
            raise ValueError("online GRPO max_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("online GRPO temperature must be non-negative")
        if self.reward_model_weight == 0 and self.exact_match_weight == 0:
            raise ValueError("online GRPO needs at least one non-zero reward source")
        for value in (self.reward_model_weight, self.exact_match_weight):
            if not math.isfinite(value):
                raise ValueError("online GRPO reward weights must be finite")


class RolloutReplayBuffer:
    """Bounded prompt-group replay buffer with JSONL persistence."""

    def __init__(self, capacity: int = 0) -> None:
        if capacity < 0:
            raise ValueError("replay capacity must be non-negative")
        self.capacity = int(capacity)
        self._items: deque[dict[str, Any]] = deque(maxlen=capacity or None)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, records: Iterable[Mapping[str, Any]]) -> None:
        for record in records:
            self._items.append(dict(record))
            if self.capacity == 0:
                # Zero means current-iteration only; retain no historical items.
                while len(self._items) > 0:
                    self._items.popleft()

    def extend_current(
        self, records: Iterable[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        current = [dict(record) for record in records]
        if self.capacity > 0:
            self.add(current)
            return list(self._items)
        return current

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in self._items:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(destination)
        return destination

    def load(self, path: str | Path) -> int:
        source = Path(path)
        if not source.exists():
            return 0
        loaded = []
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    loaded.append(payload)
        if self.capacity > 0:
            self._items.extend(loaded[-self.capacity :])
        return len(loaded)


class OnlineRolloutGenerator:
    """Generate grouped policy samples and assign reward-model/verifier rewards."""

    def __init__(
        self,
        generator: Generator,
        scorer: CompletionScorer | None,
        config: RolloutConfig,
    ) -> None:
        self.generator = generator
        self.scorer = scorer
        self.config = config
        if config.reward_model_weight != 0 and scorer is None:
            raise ValueError("reward_model_weight requires a reward scorer")

    @staticmethod
    def _expected_answer(record: Mapping[str, Any]) -> str | None:
        for key in ("expected_answer", "answer", "expected"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _exact_match(completion: str, expected: str | None) -> float:
        if expected is None:
            return 0.0
        return 1.0 if completion.strip() == expected.strip() else 0.0

    def generate_group(
        self, record: Mapping[str, Any], *, prompt_index: int = 0
    ) -> dict[str, Any] | None:
        prompt = record.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return None
        prompt = prompt.strip()
        expected = self._expected_answer(record)
        formatted = f"<|user|>\n{prompt}\n<|assistant|>\n"
        completions: list[str] = []
        attempts = 0
        max_attempts = max(self.config.group_size * 4, self.config.group_size)
        while len(completions) < self.config.group_size and attempts < max_attempts:
            seed = self.config.seed + prompt_index * 100_003 + attempts
            result = self.generator.generate(
                formatted,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                min_p=self.config.min_p,
                repetition_penalty=self.config.repetition_penalty,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                min_tokens=self.config.min_tokens,
                seed=seed,
                allow_special_tokens=True,
            )
            text = result.text.strip()
            attempts += 1
            if text and text not in completions:
                completions.append(text)
        if len(completions) < self.config.group_size:
            return None
        rewards: list[float] = []
        components: list[dict[str, float]] = []
        for completion in completions:
            rm = 0.0
            if self.scorer is not None and self.config.reward_model_weight != 0:
                rm = float(
                    self.scorer.score(
                        prompt,
                        completion,
                        normalized=self.config.normalize_reward_model,
                    )
                )
            exact = self._exact_match(completion, expected)
            total = (
                self.config.reward_model_weight * rm
                + self.config.exact_match_weight * exact
            )
            rewards.append(float(total))
            components.append(
                {"reward_model": rm, "exact_match": exact, "total": float(total)}
            )
        if max(rewards) == min(rewards):
            return None
        return {
            "prompt": prompt,
            "completions": completions,
            "rewards": rewards,
            "reward_components": components,
            "rollout": {
                "group_size": self.config.group_size,
                "seed": self.config.seed,
                "attempts": attempts,
            },
        }

    def generate(
        self, records: Iterable[Mapping[str, Any]], *, max_groups: int | None = None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        groups: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            if max_groups is not None and len(groups) >= max_groups:
                break
            try:
                group = self.generate_group(record, prompt_index=index)
                if group is None:
                    failures.append(
                        {
                            "index": index,
                            "reason": "insufficient_unique_or_nonconstant_group",
                        }
                    )
                else:
                    groups.append(group)
            except (
                Exception
            ) as error:  # rollout generation is intentionally fault-tolerant per prompt
                failures.append(
                    {
                        "index": index,
                        "reason": type(error).__name__,
                        "message": str(error),
                    }
                )
        return groups, failures


def load_prompt_records(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid rollout JSONL at {path}:{line_number}: {error}"
                    ) from error
                if isinstance(payload, str):
                    payload = {"prompt": payload}
                if isinstance(payload, dict):
                    records.append(payload)
    return records


def write_rollout_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    temporary.replace(destination)
    return destination
