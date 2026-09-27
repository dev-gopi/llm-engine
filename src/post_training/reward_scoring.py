"""Reward-model scoring, normalization and pairwise calibration utilities."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch

from tokenizer.encoder import Tokenizer

CALIBRATION_VERSION = 1


@dataclass(frozen=True)
class RewardCalibration:
    """Portable reward calibration metadata.

    ``mean``/``std`` normalize scalar rewards for downstream RL objectives.
    ``temperature`` calibrates pairwise reward differences before a sigmoid.
    The raw reward ordering is unchanged because all transforms are monotonic.
    """

    mean: float = 0.0
    std: float = 1.0
    temperature: float = 1.0
    clip: float | None = 5.0
    version: int = CALIBRATION_VERSION

    def __post_init__(self) -> None:
        if self.version != CALIBRATION_VERSION:
            raise ValueError(f"unsupported reward calibration version: {self.version}")
        if not math.isfinite(self.mean):
            raise ValueError("reward calibration mean must be finite")
        if not math.isfinite(self.std) or self.std <= 0:
            raise ValueError("reward calibration std must be positive and finite")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError(
                "reward calibration temperature must be positive and finite"
            )
        if self.clip is not None and (not math.isfinite(self.clip) or self.clip <= 0):
            raise ValueError("reward calibration clip must be positive and finite")

    def normalize(self, value: float) -> float:
        result = (float(value) - self.mean) / self.std
        if self.clip is not None:
            result = max(-self.clip, min(self.clip, result))
        return result

    def pairwise_probability(self, chosen: float, rejected: float) -> float:
        delta = (float(chosen) - float(rejected)) / self.temperature
        # Stable scalar sigmoid.
        if delta >= 0:
            z = math.exp(-delta)
            return 1.0 / (1.0 + z)
        z = math.exp(delta)
        return z / (1.0 + z)

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "RewardCalibration":
        return cls(
            mean=float(payload.get("mean", 0.0)),
            std=float(payload.get("std", 1.0)),
            temperature=float(payload.get("temperature", 1.0)),
            clip=None if payload.get("clip") is None else float(payload["clip"]),
            version=int(payload.get("version", CALIBRATION_VERSION)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RewardCalibration":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("reward calibration file must contain a JSON object")
        return cls.from_dict(payload)

    def save(self, path: str | Path, *, metadata: dict | None = None) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        if metadata:
            payload["metadata"] = dict(metadata)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return destination


def fit_reward_calibration(
    chosen_rewards: Sequence[float],
    rejected_rewards: Sequence[float],
    *,
    clip: float | None = 5.0,
) -> tuple[RewardCalibration, dict[str, float | int]]:
    """Fit normalization and a scalar pairwise temperature on held-out pairs."""
    if len(chosen_rewards) != len(rejected_rewards) or not chosen_rewards:
        raise ValueError(
            "chosen/rejected rewards must be non-empty and have equal length"
        )
    chosen = torch.tensor(chosen_rewards, dtype=torch.float64)
    rejected = torch.tensor(rejected_rewards, dtype=torch.float64)
    if not bool(torch.isfinite(chosen).all() and torch.isfinite(rejected).all()):
        raise ValueError("reward calibration inputs must be finite")
    all_scores = torch.cat((chosen, rejected))
    mean = float(all_scores.mean())
    std = float(all_scores.std(unbiased=False))
    if std < 1e-8:
        std = 1.0
    deltas = chosen - rejected
    # A bounded log-spaced search is deterministic, dependency-free and robust
    # for the small calibration sets commonly used with local reward models.
    candidates = torch.logspace(-2, 2, 161, dtype=torch.float64)
    losses = torch.stack(
        [
            torch.nn.functional.softplus(-(deltas / temperature)).mean()
            for temperature in candidates
        ]
    )
    temperature = float(candidates[int(torch.argmin(losses))])
    calibration = RewardCalibration(
        mean=mean, std=std, temperature=temperature, clip=clip
    )
    raw_accuracy = float((deltas > 0).double().mean())
    ties = float((deltas == 0).double().mean())
    probability = torch.sigmoid(deltas / temperature)
    nll = float((-torch.log(probability.clamp_min(1e-12))).mean())
    brier = float(((probability - 1.0) ** 2).mean())
    return calibration, {
        "pairs": len(chosen_rewards),
        "pairwise_accuracy": raw_accuracy,
        "tie_rate": ties,
        "mean_reward": mean,
        "std_reward": std,
        "temperature": temperature,
        "pairwise_nll": nll,
        "pairwise_brier": brier,
    }


class RewardScorer:
    """Score prompt/completion pairs using a trained ``RewardModel``."""

    def __init__(
        self,
        model,
        tokenizer: Tokenizer,
        *,
        max_length: int,
        calibration: RewardCalibration | None = None,
    ) -> None:
        if max_length < 2:
            raise ValueError("reward scorer max_length must be at least 2")
        self.model = model.eval()
        try:
            self.device = next(model.parameters()).device
        except StopIteration as error:
            raise ValueError("reward model must contain parameters") from error
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.calibration = calibration
        self.pad_id = tokenizer.token_to_id("<|pad|>")
        if self.pad_id is None:
            raise ValueError("tokenizer must define <|pad|>")

    def encode(self, prompt: str, completion: str) -> list[int]:
        prompt = prompt.strip()
        completion = completion.strip()
        if not prompt or not completion:
            raise ValueError("reward scoring requires non-empty prompt and completion")
        prefix = f"<|user|>\n{prompt}\n<|assistant|>\n"
        ids = self.tokenizer.encode(prefix, add_bos=True, allowed_special="all")
        ids += self.tokenizer.encode(completion, add_eos=True, allowed_special="all")
        if len(ids) > self.max_length:
            raise ValueError(
                f"reward-scoring sequence has {len(ids)} tokens but max_length is {self.max_length}"
            )
        return ids

    @torch.inference_mode()
    def score_many(
        self,
        pairs: Iterable[tuple[str, str]],
        *,
        batch_size: int = 8,
        normalized: bool = False,
    ) -> list[float]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        encoded = [self.encode(prompt, completion) for prompt, completion in pairs]
        scores: list[float] = []
        for start in range(0, len(encoded), batch_size):
            items = encoded[start : start + batch_size]
            width = max(len(item) for item in items)
            ids = torch.full(
                (len(items), width), self.pad_id, dtype=torch.long, device=self.device
            )
            mask = torch.zeros(
                (len(items), width), dtype=torch.bool, device=self.device
            )
            for row, item in enumerate(items):
                ids[row, : len(item)] = torch.tensor(
                    item, dtype=torch.long, device=self.device
                )
                mask[row, : len(item)] = True
            values = (
                self.model(ids, attention_mask=mask).detach().float().cpu().tolist()
            )
            scores.extend(float(value) for value in values)
        if normalized:
            if self.calibration is None:
                raise ValueError(
                    "normalized reward scoring requires calibration metadata"
                )
            return [self.calibration.normalize(value) for value in scores]
        return scores

    def score(self, prompt: str, completion: str, *, normalized: bool = False) -> float:
        return self.score_many(
            [(prompt, completion)], batch_size=1, normalized=normalized
        )[0]
