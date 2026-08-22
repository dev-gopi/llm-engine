"""Load the trained Gopi model and expose it to the serving runtime."""

from __future__ import annotations

import os
import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from pathlib import Path

from inference.generator import Generator
from inference.context import SQLiteSessionStore
from model.gpt import MiniGPT
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint
from utils.config import load_yaml
from utils.device import resolve_device
from utils.logger import get_logger

from .runtime import BackendGeneration, BackendStreamEvent, BackendUnavailableError
from .schemas import FinishReason, GenerateRequest

logger = get_logger(__name__)


class ConfiguredModelBackend:
    """Lazy backend configured through project paths or GOPI_* environment variables."""

    def __init__(
        self,
        *,
        model_config: str | Path = "configs/model.yaml",
        tokenizer_path: str | Path = "data/tokenizer",
        checkpoint_path: str | Path = "checkpoints/latest/model.pt",
        device: str = "auto",
        session_store_path: str | Path | None = None,
        system_prompt: str = "You are Gopi, a helpful assistant.",
        context_tokens: int = 1536,
    ) -> None:
        self.model_config = Path(model_config)
        self.tokenizer_path = Path(tokenizer_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.generator: Generator | None = None
        self.session_store_path = Path(session_store_path) if session_store_path else None
        self.system_prompt = system_prompt
        self.context_tokens = context_tokens
        self.sessions: SQLiteSessionStore | None = None
        self._session_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    @property
    def ready(self) -> bool:
        return self.generator is not None

    async def startup(self) -> None:
        missing = [path for path in (self.model_config, self.tokenizer_path, self.checkpoint_path) if not path.exists()]
        if missing:
            logger.warning("Model backend not loaded; missing: %s", ", ".join(map(str, missing)))
            return
        self._load()

    async def shutdown(self) -> None:
        self.generator = None

    def _load(self) -> None:
        device = resolve_device(self.device)
        tokenizer = Tokenizer.load(self.tokenizer_path)
        config = load_yaml(self.model_config)
        if int(config["vocab_size"]) != tokenizer.vocab_size:
            raise ValueError(
                f"model vocab_size={config['vocab_size']} does not match tokenizer vocab_size={tokenizer.vocab_size}"
            )
        model = MiniGPT.from_config(config, device=device)
        load_checkpoint(self.checkpoint_path, model, map_location=device, use_ema=True)
        self.generator = Generator(model, tokenizer, device=device)
        if self.session_store_path:
            self.sessions = SQLiteSessionStore(
                self.session_store_path, tokenizer,
                max_tokens=min(self.context_tokens, model.max_positions),
                system_prompt=self.system_prompt,
            )
        logger.info("Loaded model checkpoint %s on %s", self.checkpoint_path, device)

    async def generate(self, request: GenerateRequest) -> BackendGeneration:
        if request.session_id:
            async with self._session_locks[request.session_id]:
                return await self._generate_unlocked(request)
        return await self._generate_unlocked(request)

    async def _generate_unlocked(self, request: GenerateRequest) -> BackendGeneration:
        if self.generator is None:
            raise BackendUnavailableError("generation backend is not loaded")
        memory = self.sessions.load(request.session_id) if self.sessions and request.session_id else None
        options = dict(
            max_tokens=request.max_tokens, temperature=request.temperature,
            top_k=request.top_k, top_p=request.top_p,
            repetition_penalty=request.repetition_penalty, seed=request.seed, stop=request.stop,
            allow_special_tokens=bool(memory),
        )
        prompt = request.prompt
        if memory:
            memory.add("user", request.prompt)
            prompt = memory.render(add_generation_prompt=True, reserve_tokens=request.max_tokens)
        generated_ids: list[int] = []
        pieces: list[str] = []
        finish_reason = "length"
        prompt_tokens = 0
        async for event in self._stream_steps(prompt, options):
            prompt_tokens = event.prompt_tokens
            if event.token_id is not None:
                generated_ids.append(event.token_id)
            if event.token:
                pieces.append(event.token)
            if event.finish_reason is not None:
                finish_reason = event.finish_reason
        text = "".join(pieces)
        if memory and text:
            memory.add("assistant", text)
        if memory and request.session_id:
            self.sessions.save(request.session_id, memory)
        return BackendGeneration(
            text=text, prompt_tokens=prompt_tokens,
            completion_tokens=len(generated_ids), finish_reason=FinishReason(finish_reason),
        )

    async def stream(self, request: GenerateRequest) -> AsyncIterator[BackendStreamEvent]:
        if request.session_id:
            async with self._session_locks[request.session_id]:
                async for event in self._stream_unlocked(request):
                    yield event
            return
        async for event in self._stream_unlocked(request):
            yield event

    async def _stream_unlocked(self, request: GenerateRequest) -> AsyncIterator[BackendStreamEvent]:
        if self.generator is None:
            raise BackendUnavailableError("generation backend is not loaded")
        memory = self.sessions.load(request.session_id) if self.sessions and request.session_id else None
        prompt = request.prompt
        if memory:
            memory.add("user", request.prompt)
            prompt = memory.render(add_generation_prompt=True, reserve_tokens=request.max_tokens)
        generated_ids: list[int] = []
        options = dict(max_tokens=request.max_tokens, temperature=request.temperature,
                       top_k=request.top_k, top_p=request.top_p,
                       repetition_penalty=request.repetition_penalty, seed=request.seed, stop=request.stop,
                       allow_special_tokens=bool(memory))
        async for step in self._stream_steps(prompt, options):
            if step.finish_reason is not None:
                if memory and request.session_id:
                    text = self.generator.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    if text:
                        memory.add("assistant", text)
                    self.sessions.save(request.session_id, memory)
                yield BackendStreamEvent(
                    finish_reason=FinishReason(step.finish_reason),
                    prompt_tokens=step.prompt_tokens,
                    completion_tokens=step.completion_tokens,
                )
                return
            if step.token_id is not None:
                generated_ids.append(step.token_id)
            yield BackendStreamEvent(
                token=step.token, token_id=step.token_id,
                prompt_tokens=step.prompt_tokens, completion_tokens=step.completion_tokens,
            )

    async def _stream_steps(self, prompt: str, options: dict):
        for step in self.generator.stream(prompt, **options):
            yield step
            await asyncio.sleep(0)

def backend_from_environment() -> ConfiguredModelBackend:
    inference_path = Path(os.getenv("GOPI_INFERENCE_CONFIG", "configs/inference.yaml"))
    serving = load_yaml(inference_path).get("serving", {}) if inference_path.is_file() else {}
    if not isinstance(serving, dict):
        raise ValueError("inference serving configuration must be a mapping")
    return ConfiguredModelBackend(
        model_config=os.getenv("GOPI_MODEL_CONFIG", str(serving.get("model_config", "configs/model.yaml"))),
        tokenizer_path=os.getenv("GOPI_TOKENIZER_PATH", str(serving.get("tokenizer_path", "data/tokenizer"))),
        checkpoint_path=os.getenv("GOPI_CHECKPOINT_PATH", str(serving.get("checkpoint_path", "checkpoints/latest/model.pt"))),
        device=os.getenv("GOPI_DEVICE", str(serving.get("device", "auto"))),
        session_store_path=os.getenv("GOPI_SESSION_STORE", str(serving.get("session_store_path", "data/cache/sessions.sqlite"))),
        system_prompt=str(load_yaml(inference_path).get("system_prompt", "You are Gopi, a helpful assistant.")) if inference_path.is_file() else "You are Gopi, a helpful assistant.",
        context_tokens=int((load_yaml(inference_path).get("context_memory") or {}).get("max_tokens", 1536)) if inference_path.is_file() else 1536,
    )
