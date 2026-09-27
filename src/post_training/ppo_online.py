"""Online sequence-level PPO rollout helpers.

The engine's PPO core intentionally operates on sequence-level scores.  This
module owns the rollout boundary: generate one response per prompt, score it,
encode the response mask, and freeze old-policy/value statistics before PPO
updates begin.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from inference.generator import Generator
from post_training.dpo import sequence_log_probabilities
from tokenizer.encoder import Tokenizer


class CompletionScorer(Protocol):
    def score(self, prompt: str, completion: str, *, normalized: bool = False) -> float: ...


def _expected_answer(record: Mapping[str, Any]) -> str | None:
    for key in ("expected_answer", "answer", "expected"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _response_mask(total_length: int, prompt_length: int) -> Tensor:
    mask = torch.zeros(max(total_length - 1, 0), dtype=torch.bool)
    # token_ids[:, 1:] predicts positions 1..N-1. Response tokens begin at
    # prompt_length, so their prediction slots begin at prompt_length-1.
    start = max(prompt_length - 1, 0)
    if start < mask.numel():
        mask[start:] = True
    return mask


def collect_ppo_rollouts(
    records: Iterable[Mapping[str, Any]], *, generator: Generator, tokenizer: Tokenizer,
    old_policy: nn.Module, value_model: nn.Module, scorer: CompletionScorer | None,
    device: torch.device, max_sequence_length: int, max_tokens: int = 128,
    temperature: float = 0.8, top_k: int = 40, top_p: float = 1.0,
    min_p: float = 0.0, repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 3, min_tokens: int = 1,
    reward_model_weight: float = 1.0, exact_match_weight: float = 0.0,
    normalize_reward_model: bool = False, seed: int = 42,
    max_rollouts: int | None = None,
) -> tuple[dict[str, Tensor], list[dict[str, Any]]]:
    """Generate and freeze a PPO rollout batch.

    Invalid/empty/over-length samples are skipped and described in the returned
    failure list.  At least one non-zero reward source is required.
    """
    if max_sequence_length < 2 or max_tokens < 1:
        raise ValueError("invalid PPO rollout lengths")
    if reward_model_weight == 0 and exact_match_weight == 0:
        raise ValueError("PPO needs at least one non-zero reward source")
    if reward_model_weight != 0 and scorer is None:
        raise ValueError("reward_model_weight requires a reward scorer")
    pad_id = tokenizer.token_to_id("<|pad|>")
    if pad_id is None:
        raise ValueError("tokenizer must define <|pad|>")

    samples: list[tuple[list[int], Tensor, float, str, str]] = []
    failures: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if max_rollouts is not None and len(samples) >= max_rollouts:
            break
        prompt = record.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            failures.append({"index": index, "reason": "missing_prompt"})
            continue
        prompt = prompt.strip()
        formatted = f"<|user|>\n{prompt}\n<|assistant|>\n"
        try:
            result = generator.generate(
                formatted, max_tokens=max_tokens, temperature=temperature,
                top_k=top_k, top_p=top_p, min_p=min_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size, min_tokens=min_tokens,
                seed=seed + index * 100_003, allow_special_tokens=True,
            )
            completion = result.text.strip()
            if not completion:
                failures.append({"index": index, "reason": "empty_completion"})
                continue
            prefix_ids = tokenizer.encode(formatted, add_bos=True, allowed_special="all")
            ids = prefix_ids + tokenizer.encode(completion, add_eos=True, allowed_special="all")
            if len(ids) < 2 or len(ids) > max_sequence_length:
                failures.append({"index": index, "reason": "sequence_too_long", "tokens": len(ids)})
                continue
            rm = 0.0 if scorer is None else float(scorer.score(prompt, completion, normalized=normalize_reward_model))
            expected = _expected_answer(record)
            exact = 1.0 if expected is not None and completion.strip() == expected else 0.0
            reward = reward_model_weight * rm + exact_match_weight * exact
            if not torch.isfinite(torch.tensor(reward)):
                failures.append({"index": index, "reason": "non_finite_reward"})
                continue
            samples.append((ids, _response_mask(len(ids), len(prefix_ids)), float(reward), prompt, completion))
        except Exception as error:  # rollout faults are isolated per prompt
            failures.append({"index": index, "reason": type(error).__name__, "message": str(error)})

    if not samples:
        raise ValueError("PPO rollout generation produced no usable samples")
    width = max(len(item[0]) for item in samples)
    token_ids = torch.full((len(samples), width), int(pad_id), dtype=torch.long, device=device)
    mask = torch.zeros((len(samples), width - 1), dtype=torch.bool, device=device)
    attention = torch.zeros((len(samples), width), dtype=torch.bool, device=device)
    rewards = torch.tensor([item[2] for item in samples], dtype=torch.float32, device=device)
    for row, (ids, response_mask, *_rest) in enumerate(samples):
        n = len(ids)
        token_ids[row, :n] = torch.tensor(ids, dtype=torch.long, device=device)
        mask[row, : response_mask.numel()] = response_mask.to(device)
        attention[row, :n] = True

    old_policy.eval(); value_model.eval()
    with torch.no_grad():
        output = old_policy(token_ids, attention_mask=attention)
        logits = output[0] if isinstance(output, tuple) else output
        old_log_probs = sequence_log_probabilities(logits, token_ids, mask)
        old_values = value_model(token_ids, attention_mask=attention)
    advantages = rewards - old_values
    returns = rewards
    rollout = {
        "token_ids": token_ids,
        "mask": mask,
        "old_log_probs": old_log_probs.detach(),
        "old_values": old_values.detach(),
        "advantages": advantages.detach(),
        "returns": returns.detach(),
        "rewards": rewards.detach(),
    }
    return rollout, failures
