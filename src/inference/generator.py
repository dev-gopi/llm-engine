"""Autoregressive text generation connecting tokenizer, model, and sampler."""

from __future__ import annotations

from dataclasses import dataclass
import itertools

import torch
from torch import nn
import torch.nn.functional as F

from tokenizer.encoder import Tokenizer
from utils.device import resolve_device
from utils.logger import get_logger

from .sampler import TopKSampler
from .context import ConversationMemory
from .kv_cache import KVCache
from .paged_kv_cache import PagedKVCache, PagedPrefixCache, PrefixCache

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


@dataclass
class BatchedGenerationState:
    prompt_ids: list[int]
    all_ids: list[int]
    generated: list[int]
    logits: torch.Tensor
    cache: tuple
    cache_mask: torch.Tensor
    random: torch.Generator
    options: dict
    emitted_text: str = ""
    steps: int = 0
    page_request_id: str | None = None


class Generator:
    """Generate text from a decoder-only model using bounded context windows."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        *,
        device: str | torch.device = "auto",
        prefix_cache_capacity: int = 0,
        paged_kv_pages: int = 0,
        paged_kv_page_size: int = 16,
    ) -> None:
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.sampler = TopKSampler()
        self.max_positions = int(getattr(model, "max_positions", 0))
        if self.max_positions < 1:
            raise ValueError("model must expose a positive max_positions value")
        self.eos_token_id = tokenizer.token_to_id("<|eos|>")
        self.prefix_cache: PrefixCache | PagedPrefixCache | None = None
        self.paged_kv_allocator: PagedKVCache | None = None
        self._paged_request_ids = itertools.count(1)
        self.prefix_cache_hits = 0
        self.prefix_cache_misses = 0
        if paged_kv_pages:
            first_attention = getattr(model, "blocks", [None])[0].attn
            allocator = PagedKVCache(
                num_pages=paged_kv_pages, page_size=paged_kv_page_size,
                layers=len(model.blocks), kv_heads=first_attention.kv_heads,
                head_dim=first_attention.head_dim, device=self.device,
                dtype=next(model.parameters()).dtype,
            )
            self.paged_kv_allocator = allocator
            if prefix_cache_capacity:
                self.prefix_cache = PagedPrefixCache(
                    allocator, capacity=prefix_cache_capacity
                )
        elif prefix_cache_capacity:
            self.prefix_cache = PrefixCache(prefix_cache_capacity)

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

        logits, raw_cache = self._prefill(prompt_ids)
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
    def start_batched_stream(self, prompt: str, **options) -> BatchedGenerationState:
        """Prefill one request for admission to a token-level decode scheduler."""
        prompt_ids = self.tokenizer.encode(
            prompt, add_bos=True,
            allowed_special="all" if options.get("allow_special_tokens", False) else (),
        )
        if not prompt_ids or len(prompt_ids) >= self.max_positions:
            raise ValueError("prompt is empty or exceeds the model context")
        maximum = int(options.get("max_tokens", 128))
        if maximum < 1:
            raise ValueError("max_tokens must be positive")
        random = torch.Generator(device=self.device)
        if options.get("seed") is not None:
            random.manual_seed(int(options["seed"]))
        logits, cache = self._prefill(prompt_ids)
        state = BatchedGenerationState(
            prompt_ids=list(prompt_ids), all_ids=list(prompt_ids), generated=[],
            logits=logits, cache=cache,
            cache_mask=torch.ones(len(prompt_ids), dtype=torch.bool, device=self.device),
            random=random, options=dict(options),
        )
        if self.paged_kv_allocator is not None:
            request_id = f"active-{next(self._paged_request_ids)}"
            capacity = min(self.max_positions, len(prompt_ids) + maximum)
            self.paged_kv_allocator.reserve(request_id, capacity)
            keys = torch.stack([layer[0].squeeze(0) for layer in cache])
            values = torch.stack([layer[1].squeeze(0) for layer in cache])
            self.paged_kv_allocator.append(request_id, keys, values)
            state.page_request_id = request_id
            state.cache = self._materialize_active_cache(request_id)
        return state

    @torch.inference_mode()
    def decode_batched_stream(
        self, states: list[BatchedGenerationState]
    ) -> list[tuple[GenerationStep, bool]]:
        """Sample one token per state and run one model call for all survivors."""
        if not states:
            return []
        results: list[tuple[GenerationStep, bool] | None] = [None] * len(states)
        survivors: list[tuple[int, BatchedGenerationState, int]] = []
        for index, state in enumerate(states):
            options = state.options
            logits = state.logits[:, -1, :].clone()
            penalty = float(options.get("repetition_penalty", 1.0))
            if penalty <= 0:
                raise ValueError("repetition_penalty must be positive")
            self._apply_repetition_penalty(logits, set(state.all_ids), penalty)
            token_id = int(self.sampler(
                logits, temperature=float(options.get("temperature", .8)),
                top_k=int(options.get("top_k", 40)), top_p=float(options.get("top_p", 1.0)),
                generator=state.random,
            ).item())
            state.steps += 1
            eos = self.eos_token_id is not None and token_id == self.eos_token_id
            if not eos:
                state.generated.append(token_id)
                state.all_ids.append(token_id)
            text = self.tokenizer.decode(state.generated, skip_special_tokens=True)
            stops = options.get("stop") or []
            positions = [text.find(value) for value in stops if value in text]
            stopped = bool(positions)
            visible = text[:min(positions)] if stopped else text
            delta = visible[len(state.emitted_text):] if visible.startswith(state.emitted_text) else ""
            state.emitted_text = visible
            limit = min(int(options.get("max_tokens", 128)), self.max_positions - len(state.prompt_ids))
            done = eos or stopped or state.steps >= limit
            finish = "stop" if eos or stopped else ("length" if done else None)
            results[index] = (GenerationStep(
                delta, None if eos else token_id, len(state.prompt_ids),
                len(state.generated), finish,
            ), done)
            if not done:
                survivors.append((index, state, token_id))

        if survivors:
            for _, state, _ in survivors:
                if state.page_request_id is not None:
                    state.cache = self._materialize_active_cache(state.page_request_id)
            maximum_cache = max(state.cache[0][0].shape[2] for _, state, _ in survivors)
            masks, positions, batched_layers = [], [], []
            for _, state, _ in survivors:
                padding = maximum_cache - state.cache[0][0].shape[2]
                masks.append(torch.cat((
                    torch.zeros(padding, dtype=torch.bool, device=self.device),
                    state.cache_mask,
                    torch.ones(1, dtype=torch.bool, device=self.device),
                )))
                positions.append(int(state.cache_mask.sum()))
            for layer in range(len(survivors[0][1].cache)):
                keys, values = [], []
                for _, state, _ in survivors:
                    key, value = state.cache[layer]
                    padding = maximum_cache - key.shape[2]
                    keys.append(F.pad(key, (0, 0, padding, 0)))
                    values.append(F.pad(value, (0, 0, padding, 0)))
                batched_layers.append((torch.cat(keys), torch.cat(values)))
            output = self.model(
                torch.tensor([token for _, _, token in survivors], device=self.device).unsqueeze(1),
                attention_mask=torch.stack(masks),
                position_ids=torch.tensor(positions, device=self.device).unsqueeze(1),
                past_key_values=tuple(batched_layers), use_cache=True,
            )
            if not isinstance(output, tuple):
                raise RuntimeError("model did not return a requested KV cache")
            logits, cache = output
            for row, (_, state, _) in enumerate(survivors):
                state.logits = logits[row:row + 1]
                if state.page_request_id is not None:
                    keys = torch.stack([key[row, :, -1:, :] for key, _ in cache])
                    values = torch.stack([value[row, :, -1:, :] for _, value in cache])
                    self.paged_kv_allocator.append(state.page_request_id, keys, values)
                    state.cache = self._materialize_active_cache(state.page_request_id)
                    state.cache_mask = torch.ones(
                        state.cache[0][0].shape[2], dtype=torch.bool, device=self.device
                    )
                else:
                    state.cache = tuple((key[row:row + 1], value[row:row + 1]) for key, value in cache)
                    state.cache_mask = torch.stack(masks)[row]
        return [result for result in results if result is not None]

    def release_batched_stream(self, state: BatchedGenerationState) -> None:
        if state.page_request_id is not None and self.paged_kv_allocator is not None:
            request_id, state.page_request_id = state.page_request_id, None
            if request_id in self.paged_kv_allocator.tables:
                self.paged_kv_allocator.release(request_id)
        state.cache = ()

    def _materialize_active_cache(self, request_id: str) -> tuple:
        if self.paged_kv_allocator is None:
            raise RuntimeError("paged KV allocator is not configured")
        keys, values = self.paged_kv_allocator.materialize(request_id)
        return tuple(
            (keys[layer].unsqueeze(0), values[layer].unsqueeze(0))
            for layer in range(keys.shape[0])
        )

    @torch.inference_mode()
    def generate_batch(self, prompts: list[str], **options) -> list[GenerationResult]:
        """Decode prompt cohorts in tensor batches and compact completed KV rows.

        Prompts of equal token length share prefill and decode calls. Different
        lengths form independent cohorts, avoiding padding tokens in cached
        attention while still allowing each cohort to shrink continuously.
        """
        if not prompts:
            return []
        max_tokens = int(options.get("max_tokens", 128))
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        penalty = float(options.get("repetition_penalty", 1.0))
        if penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        encoded = [self.tokenizer.encode(
            prompt, add_bos=True,
            allowed_special="all" if options.get("allow_special_tokens", False) else (),
        ) for prompt in prompts]
        if any(not ids or len(ids) >= self.max_positions for ids in encoded):
            raise ValueError("a prompt is empty or exceeds the model context")
        results: list[GenerationResult | None] = [None] * len(prompts)
        cohorts: dict[int, list[int]] = {}
        for index, ids in enumerate(encoded):
            cohorts.setdefault(len(ids), []).append(index)

        for prompt_length, indexes in cohorts.items():
            active = list(indexes)
            all_ids = [list(encoded[i]) for i in active]
            generated: list[list[int]] = [[] for _ in active]
            randoms = [torch.Generator(device=self.device) for _ in active]
            seed = options.get("seed")
            if seed is not None:
                for offset, random in enumerate(randoms):
                    random.manual_seed(int(seed) + indexes[offset])
            output = self.model(torch.tensor([encoded[i] for i in active], device=self.device), use_cache=True)
            if not isinstance(output, tuple):
                raise RuntimeError("model did not return a requested KV cache")
            logits, cache = output
            limit = min(max_tokens, self.max_positions - prompt_length)
            for step in range(limit):
                survivors: list[int] = []
                next_tokens: list[int] = []
                for row, original_index in enumerate(active):
                    row_logits = logits[row:row + 1, -1, :].clone()
                    self._apply_repetition_penalty(row_logits, set(all_ids[row]), penalty)
                    token_id = int(self.sampler(
                        row_logits, temperature=float(options.get("temperature", .8)),
                        top_k=int(options.get("top_k", 40)), top_p=float(options.get("top_p", 1.0)),
                        generator=randoms[row],
                    ).item())
                    eos = self.eos_token_id is not None and token_id == self.eos_token_id
                    if not eos:
                        generated[row].append(token_id)
                        all_ids[row].append(token_id)
                    text = self.tokenizer.decode(generated[row], skip_special_tokens=True)
                    stopped = any(value in text for value in (options.get("stop") or []))
                    done = eos or stopped or step + 1 == limit
                    if done:
                        results[original_index] = GenerationResult(
                            self._trim_stop(text, options.get("stop") or []), tuple(generated[row]),
                            prompt_length, "stop" if eos or stopped else "length",
                        )
                    else:
                        survivors.append(row)
                        next_tokens.append(token_id)
                if not survivors:
                    break
                select = torch.tensor(survivors, device=self.device)
                cache = tuple((key.index_select(0, select), value.index_select(0, select)) for key, value in cache)
                active = [active[row] for row in survivors]
                all_ids = [all_ids[row] for row in survivors]
                generated = [generated[row] for row in survivors]
                randoms = [randoms[row] for row in survivors]
                output = self.model(torch.tensor(next_tokens, device=self.device).unsqueeze(1),
                                    past_key_values=cache, use_cache=True)
                if not isinstance(output, tuple):
                    raise RuntimeError("model did not return a requested KV cache")
                logits, cache = output
        if any(result is None for result in results):
            raise RuntimeError("batched generation did not finalize every request")
        return [result for result in results if result is not None]

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
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
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
        logits, raw_cache = self._prefill(prompt_ids)
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

    def _prefill(self, prompt_ids: list[int]):
        key = tuple(prompt_ids)
        if self.prefix_cache is not None:
            cached = self.prefix_cache.get(key)
            if cached is not None:
                self.prefix_cache_hits += 1
                logits, raw_cache = cached
                if isinstance(self.prefix_cache, PrefixCache):
                    logits, raw_cache = cached
                return logits, raw_cache
            self.prefix_cache_misses += 1
        output = self.model(torch.tensor([prompt_ids], dtype=torch.long, device=self.device), use_cache=True)
        if not isinstance(output, tuple):
            raise RuntimeError("model did not return a requested KV cache")
        logits, raw_cache = output
        if self.prefix_cache is not None:
            value = (logits.detach().clone(), tuple(
                (key.detach().clone(), value.detach().clone()) for key, value in raw_cache
            ))
            if isinstance(self.prefix_cache, PagedPrefixCache):
                self.prefix_cache.put(key, logits, raw_cache)
            else:
                self.prefix_cache.put(key, value)
        return logits, raw_cache
