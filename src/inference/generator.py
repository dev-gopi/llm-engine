"""Autoregressive text generation connecting tokenizer, model, and sampler."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tokenizer.encoder import Tokenizer
from utils.device import resolve_device
from utils.logger import get_logger

from .sampler import TopKSampler
from .context import ConversationMemory
from .kv_cache import KVCache

logger = get_logger(__name__)


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: tuple[int, ...]
    prompt_tokens: int
    finish_reason: str


@dataclass(frozen=True)
class GenerationStep:
    token: str
    token_id: int | None
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None = None


class Generator:
    """Generate text from a decoder-only model using bounded context windows."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        *,
        device: str | torch.device = "auto",
    ) -> None:
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.sampler = TopKSampler()
        self.max_positions = int(getattr(model, "max_positions", 0))
        if self.max_positions < 1:
            raise ValueError("model must expose a positive max_positions value")
        self.eos_token_id = tokenizer.token_to_id("<|eos|>")

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        seed: int | None = None,
        stop: list[str] | None = None,
        allow_special_tokens: bool = False,
    ) -> GenerationResult:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        prompt_ids = self.tokenizer.encode(
            prompt, add_bos=True, allowed_special="all" if allow_special_tokens else ()
        )
        if not prompt_ids:
            raise ValueError("prompt encoded to no tokens")
        if len(prompt_ids) >= self.max_positions:
            raise ValueError(
                f"prompt has {len(prompt_ids)} tokens but model context is {self.max_positions}"
            )

        random = torch.Generator(device=self.device)
        if seed is not None:
            random.manual_seed(seed)
        all_ids = list(prompt_ids)
        generated: list[int] = []
        finish_reason = "length"
        limit = min(max_tokens, self.max_positions - len(prompt_ids))
        stop_sequences = stop or []

        input_ids = torch.tensor([all_ids], dtype=torch.long, device=self.device)
        model_output = self.model(input_ids, use_cache=True)
        if not isinstance(model_output, tuple):
            raise RuntimeError("model did not return a requested KV cache")
        logits, raw_cache = model_output
        cache = KVCache(raw_cache)
        for _ in range(limit):
            next_logits = logits[:, -1, :]
            self._apply_repetition_penalty(next_logits, set(all_ids), repetition_penalty)
            next_id = int(
                self.sampler(
                    next_logits, temperature=temperature, top_k=top_k,
                    top_p=top_p, generator=random,
                ).item()
            )
            if self.eos_token_id is not None and next_id == self.eos_token_id:
                finish_reason = "stop"
                break
            generated.append(next_id)
            all_ids.append(next_id)
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            if any(sequence in text for sequence in stop_sequences):
                finish_reason = "stop"
                text = self._trim_stop(text, stop_sequences)
                return GenerationResult(text, tuple(generated), len(prompt_ids), finish_reason)
            step_input = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
            model_output = self.model(step_input, past_key_values=cache.values, use_cache=True)
            if not isinstance(model_output, tuple):
                raise RuntimeError("model did not return a requested KV cache")
            logits, raw_cache = model_output
            cache.update(raw_cache)

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        logger.debug("Generated %d tokens from a %d-token prompt", len(generated), len(prompt_ids))
        return GenerationResult(text, tuple(generated), len(prompt_ids), finish_reason)

    def generate_chat(
        self,
        memory: ConversationMemory,
        user_message: str,
        **generation_options,
    ) -> GenerationResult:
        """Add a user turn, generate from bounded history, then remember the reply."""
        memory.add("user", user_message)
        reserve = int(generation_options.get("max_tokens", 128))
        prompt = memory.render(add_generation_prompt=True, reserve_tokens=reserve)
        result = self.generate(prompt, allow_special_tokens=True, **generation_options)
        if result.text:
            memory.add("assistant", result.text)
        return result

    @torch.inference_mode()
    def stream(
        self,
        prompt: str,
        **options,
    ):
        """Yield tokens as each decoding step completes, followed by a final event."""
        max_tokens = int(options.get("max_tokens", 128))
        temperature = float(options.get("temperature", 0.8))
        top_k = int(options.get("top_k", 40))
        top_p = float(options.get("top_p", 1.0))
        repetition_penalty = float(options.get("repetition_penalty", 1.0))
        stop_sequences = options.get("stop") or []
        prompt_ids = self.tokenizer.encode(
            prompt, add_bos=True,
            allowed_special="all" if options.get("allow_special_tokens", False) else (),
        )
        if not prompt_ids or len(prompt_ids) >= self.max_positions:
            raise ValueError("prompt is empty or exceeds the model context")
        random = torch.Generator(device=self.device)
        if options.get("seed") is not None:
            random.manual_seed(int(options["seed"]))
        all_ids, generated = list(prompt_ids), []
        emitted_text = ""
        output = self.model(torch.tensor([all_ids], device=self.device), use_cache=True)
        if not isinstance(output, tuple):
            raise RuntimeError("model did not return a requested KV cache")
        logits, raw_cache = output
        cache = KVCache(raw_cache)
        finish_reason = "length"
        for _ in range(min(max_tokens, self.max_positions - len(prompt_ids))):
            next_logits = logits[:, -1, :]
            self._apply_repetition_penalty(next_logits, set(all_ids), repetition_penalty)
            token_id = int(self.sampler(next_logits, temperature=temperature, top_k=top_k, top_p=top_p, generator=random).item())
            if self.eos_token_id is not None and token_id == self.eos_token_id:
                finish_reason = "stop"
                break
            generated.append(token_id)
            all_ids.append(token_id)
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            stop_positions = [text.find(sequence) for sequence in stop_sequences if sequence in text]
            stopped = bool(stop_positions)
            visible = text[: min(stop_positions)] if stopped else text
            if not stopped and stop_sequences:
                visible = visible[: max(0, len(visible) - max(map(len, stop_sequences)) + 1)]
            visible = visible.rstrip("\ufffd")
            delta = visible[len(emitted_text) :] if visible.startswith(emitted_text) else ""
            emitted_text = visible
            yield GenerationStep(delta, token_id, len(prompt_ids), len(generated))
            if stopped:
                finish_reason = "stop"
                break
            output = self.model(
                torch.tensor([[token_id]], device=self.device),
                past_key_values=cache.values,
                use_cache=True,
            )
            if not isinstance(output, tuple):
                raise RuntimeError("model did not return a requested KV cache")
            logits, raw_cache = output
            cache.update(raw_cache)
        final_text = self.tokenizer.decode(generated, skip_special_tokens=True)
        stop_positions = [final_text.find(sequence) for sequence in stop_sequences if sequence in final_text]
        if stop_positions:
            final_text = final_text[: min(stop_positions)]
        remaining = final_text[len(emitted_text) :] if final_text.startswith(emitted_text) else ""
        if remaining:
            yield GenerationStep(remaining, None, len(prompt_ids), len(generated))
        yield GenerationStep("", None, len(prompt_ids), len(generated), finish_reason)

    @staticmethod
    def _apply_repetition_penalty(logits: torch.Tensor, used: set[int], penalty: float) -> None:
        if penalty == 1.0 or not used:
            return
        indices = torch.tensor(sorted(used), device=logits.device)
        selected = logits[:, indices]
        logits[:, indices] = torch.where(selected < 0, selected * penalty, selected / penalty)

    @staticmethod
    def _trim_stop(text: str, stop_sequences: list[str]) -> str:
        endings = [text.find(sequence) for sequence in stop_sequences if sequence in text]
        return text[: min(endings)] if endings else text
