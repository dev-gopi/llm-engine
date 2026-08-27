"""Load the trained Gopi model and expose it to the serving runtime."""

from __future__ import annotations

import os
import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from pathlib import Path

from inference.generator import Generator
from inference.context import SQLiteSessionStore, format_system_prompt
from inference.local_tools import direct_tool_answer, tool_context
from inference.prompt_safety import blocked_prompt_message
from inference.web_search import build_search_prompt, format_sources, search_brave, search_searxng
from datasets.preprocessor import format_messages
from model.gpt import MiniGPT
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint
from utils.config import load_yaml
from utils.device import resolve_device
from utils.logger import get_logger

from .runtime import BackendGeneration, BackendStreamEvent, BackendUnavailableError
from .schemas import FinishReason, GenerateRequest

logger = get_logger(__name__)

DEFAULT_EMPTY_RESPONSE = "Sorry, I couldn't generate a response. Please try rephrasing your prompt."
DEFAULT_NO_RESULTS = "Sorry, I couldn't find any results for that search."
COMPACT_SAFETY_PROMPT = "Be safe. Refuse harm."


class ConfiguredModelBackend:
    """Lazy backend configured through project paths or GOPI_* environment variables."""

    def __init__(
        self,
        *,
        model_config: str | Path = "configs/model.cpu.yaml",
        tokenizer_path: str | Path = "data/tokenizer",
        checkpoint_path: str | Path = "checkpoints/latest/model.pt",
        device: str = "auto",
        session_store_path: str | Path | None = None,
        system_prompt: str = "You are Gopi, a helpful assistant.",
        response_format: str | None = None,
        context_tokens: int = 1536,
        web_search: dict | None = None,
    ) -> None:
        self.model_config = Path(model_config)
        self.tokenizer_path = Path(tokenizer_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.generator: Generator | None = None
        self.session_store_path = Path(session_store_path) if session_store_path else None
        self.system_prompt = system_prompt
        self.response_format = response_format
        self.context_tokens = context_tokens
        self.web_search = web_search or {}
        self.sessions: SQLiteSessionStore | None = None
        self._session_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    @property
    def ready(self) -> bool:
        return self.generator is not None

    async def startup(self) -> None:
        if not self.checkpoint_path.exists():
            # Search for candidates by priority and modification time
            candidates = [
                Path("checkpoints/finetuning/best.pt"),
                Path("checkpoints/finetuning/latest.pt"),
                Path("checkpoints/latest/model.pt"),
                Path("checkpoints/latest/best.pt"),
                Path("checkpoints/pretraining/best.pt"),
            ]
            # Also find all .pt files under checkpoints/ sorted by modification time
            all_pt_files = sorted(Path("checkpoints").glob("**/*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            for file in all_pt_files:
                if file not in candidates:
                    candidates.append(file)

            for fallback in candidates:
                if fallback.exists():
                    logger.info("Configured checkpoint %s not found; falling back to latest checkpoint %s", self.checkpoint_path, fallback)
                    self.checkpoint_path = fallback
                    break

        missing = [path for path in (self.model_config, self.tokenizer_path, self.checkpoint_path) if not path.exists()]
        if missing:
            logger.warning("Model backend not loaded; missing: %s", ", ".join(map(str, missing)))
            return
        try:
            self._load()
        except (OSError, RuntimeError, TypeError, ValueError):
            logger.exception(
                "Model backend not loaded; verify that the model configuration, "
                "tokenizer, and checkpoint are compatible"
            )
            self.generator = None

    async def shutdown(self) -> None:
        self.generator = None

    def _load(self) -> None:
        device = resolve_device(self.device)
        tokenizer = Tokenizer.load(self.tokenizer_path)
        config = load_yaml(self.model_config)
        if int(config["vocab_size"]) != tokenizer.vocab_size:
            config["vocab_size"] = tokenizer.vocab_size

        try:
            model = MiniGPT.from_config(config, device=device)
            load_checkpoint(self.checkpoint_path, model, map_location=device, use_ema=True)
        except RuntimeError as error:
            # Fallback to alternative model configs (e.g. model.gpu.yaml) if architecture mismatch occurs
            alt_config_path = Path("configs/model.gpu.yaml") if self.model_config.name == "model.cpu.yaml" else Path("configs/model.cpu.yaml")
            if alt_config_path.exists():
                logger.info("Retrying checkpoint load with alternate config: %s", alt_config_path)
                alt_config = load_yaml(alt_config_path)
                if int(alt_config["vocab_size"]) != tokenizer.vocab_size:
                    alt_config["vocab_size"] = tokenizer.vocab_size
                model = MiniGPT.from_config(alt_config, device=device)
                load_checkpoint(self.checkpoint_path, model, map_location=device, use_ema=True)
                self.model_config = alt_config_path
                config = alt_config
            else:
                raise error

        self.generator = Generator(model, tokenizer, device=device)
        if self.session_store_path:
            self.sessions = SQLiteSessionStore(
                self.session_store_path, tokenizer,
                max_tokens=min(self.context_tokens, model.max_positions),
                system_prompt=self.system_prompt,
            )
        logger.info("Successfully loaded model checkpoint %s using config %s on %s", self.checkpoint_path, self.model_config, device)

    async def generate(self, request: GenerateRequest) -> BackendGeneration:
        if request.session_id:
            async with self._session_locks[request.session_id]:
                return await self._generate_unlocked(request)
        return await self._generate_unlocked(request)

    async def _generate_unlocked(self, request: GenerateRequest) -> BackendGeneration:
        if self.generator is None:
            raise BackendUnavailableError("generation backend is not loaded")
        safety_refusal = blocked_prompt_message(request.prompt)
        if safety_refusal is not None:
            prompt_tokens = len(self.generator.tokenizer.encode(request.prompt, add_bos=True))
            completion_tokens = len(self.generator.tokenizer.encode(safety_refusal))
            return BackendGeneration(
                text=safety_refusal, prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens, finish_reason=FinishReason.STOP,
            )
        direct_answer = direct_tool_answer(request.prompt, request.tools)
        if direct_answer is not None:
            prompt_tokens = len(self.generator.tokenizer.encode(request.prompt, add_bos=True))
            completion_tokens = len(self.generator.tokenizer.encode(direct_answer))
            return BackendGeneration(
                text=direct_answer, prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens, finish_reason=FinishReason.STOP,
            )
        memory = self.sessions.load(request.session_id) if self.sessions and request.session_id else None
        user_prompt, search_results = await ConfiguredModelBackend._prepare_user_prompt(self, request)
        if user_prompt is None:
            prompt_tokens = len(self.generator.tokenizer.encode(request.prompt, add_bos=True))
            completion_tokens = len(self.generator.tokenizer.encode(DEFAULT_NO_RESULTS))
            return BackendGeneration(
                text=DEFAULT_NO_RESULTS, prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens, finish_reason=FinishReason.STOP,
            )
        options = dict(
            max_tokens=request.max_tokens, temperature=request.temperature,
            top_k=request.top_k, top_p=request.top_p,
            repetition_penalty=request.repetition_penalty, seed=request.seed, stop=request.stop,
            allow_special_tokens=True,
        )
        response_format = request.response_format or getattr(self, "response_format", None)
        system_prompt = format_system_prompt(self.system_prompt, response_format, request.mode)
        prompt = ConfiguredModelBackend._format_new_conversation(self, system_prompt, user_prompt)
        if memory:
            memory.set_system_prompt(system_prompt)
            memory.add("user", user_prompt)
            prompt = memory.render(add_generation_prompt=True, reserve_tokens=request.max_tokens)
        prompt_ids = self.generator.tokenizer.encode(prompt, add_bos=True, allowed_special="all")
        logger.debug("Generating from a %d-token prompt", len(prompt_ids))
        generated_ids: list[int] = []
        pieces: list[str] = []
        finish_reason = "length"
        prompt_tokens = len(prompt_ids)
        async for event in self._stream_steps(prompt, options):
            prompt_tokens = event.prompt_tokens
            if event.token_id is not None:
                generated_ids.append(event.token_id)
            if event.token:
                pieces.append(event.token)
            if event.finish_reason is not None:
                finish_reason = event.finish_reason
        text = "".join(pieces).strip()
        if not text:
            text = DEFAULT_EMPTY_RESPONSE
            generated_ids = list(self.generator.tokenizer.encode(text))
        if memory:
            memory.add("assistant", text)
        if memory and request.session_id:
            self.sessions.save(request.session_id, memory)
        visible_text = text
        if search_results:
            visible_text = f"{text.rstrip()}\n\n{format_sources(search_results)}".strip()
        return BackendGeneration(
            text=visible_text, prompt_tokens=prompt_tokens,
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
        safety_refusal = blocked_prompt_message(request.prompt)
        if safety_refusal is not None:
            prompt_tokens = len(self.generator.tokenizer.encode(request.prompt, add_bos=True))
            completion_tokens = len(self.generator.tokenizer.encode(safety_refusal))
            yield BackendStreamEvent(
                token=safety_refusal, prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            yield BackendStreamEvent(
                finish_reason=FinishReason.STOP, prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            return
        direct_answer = direct_tool_answer(request.prompt, request.tools)
        if direct_answer is not None:
            prompt_tokens = len(self.generator.tokenizer.encode(request.prompt, add_bos=True))
            completion_tokens = len(self.generator.tokenizer.encode(direct_answer))
            yield BackendStreamEvent(
                token=direct_answer, prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            yield BackendStreamEvent(
                finish_reason=FinishReason.STOP, prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            return
        memory = self.sessions.load(request.session_id) if self.sessions and request.session_id else None
        user_prompt, search_results = await ConfiguredModelBackend._prepare_user_prompt(self, request)
        if user_prompt is None:
            prompt_tokens = len(self.generator.tokenizer.encode(request.prompt, add_bos=True))
            completion_tokens = len(self.generator.tokenizer.encode(DEFAULT_NO_RESULTS))
            yield BackendStreamEvent(
                token=DEFAULT_NO_RESULTS, prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            yield BackendStreamEvent(
                finish_reason=FinishReason.STOP, prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            return
        response_format = request.response_format or getattr(self, "response_format", None)
        system_prompt = format_system_prompt(self.system_prompt, response_format, request.mode)
        prompt = ConfiguredModelBackend._format_new_conversation(self, system_prompt, user_prompt)
        if memory:
            memory.set_system_prompt(system_prompt)
            memory.add("user", user_prompt)
            prompt = memory.render(add_generation_prompt=True, reserve_tokens=request.max_tokens)
        generated_ids: list[int] = []
        pieces: list[str] = []
        options = dict(max_tokens=request.max_tokens, temperature=request.temperature,
                       top_k=request.top_k, top_p=request.top_p,
                       repetition_penalty=request.repetition_penalty, seed=request.seed, stop=request.stop,
                       allow_special_tokens=True)
        async for step in self._stream_steps(prompt, options):
            if step.finish_reason is not None:
                response_text = "".join(pieces).strip()
                if not response_text:
                    response_text = DEFAULT_EMPTY_RESPONSE
                    fallback_tokens = len(self.generator.tokenizer.encode(response_text))
                    yield BackendStreamEvent(token=response_text, completion_tokens=fallback_tokens)
                if memory and request.session_id:
                    memory.add("assistant", response_text)
                    self.sessions.save(request.session_id, memory)
                if search_results:
                    yield BackendStreamEvent(token=f"\n\n{format_sources(search_results)}")
                yield BackendStreamEvent(
                    finish_reason=FinishReason(step.finish_reason),
                    prompt_tokens=step.prompt_tokens,
                    completion_tokens=(
                        len(self.generator.tokenizer.encode(response_text))
                        if not generated_ids else step.completion_tokens
                    ),
                )
                return
            if step.token_id is not None:
                generated_ids.append(step.token_id)
            if step.token:
                pieces.append(step.token)
            yield BackendStreamEvent(
                token=step.token, token_id=step.token_id,
                prompt_tokens=step.prompt_tokens, completion_tokens=step.completion_tokens,
            )

    async def _prepare_user_prompt(self, request: GenerateRequest):
        raw_prompt = request.prompt.strip()
        slash_calculator = raw_prompt.lower().startswith(("/calculate ", "/calc "))
        slash_datetime = raw_prompt.lower() in {"/time", "/date", "/datetime"}
        selected_tools = list(request.tools)
        if slash_calculator and "calculator" not in selected_tools:
            selected_tools.append("calculator")
        if slash_datetime and "datetime" not in selected_tools:
            selected_tools.append("datetime")
        slash_search = raw_prompt.lower() == "/search" or raw_prompt.lower().startswith("/search ")
        if not request.web_search and not slash_search:
            return tool_context(raw_prompt, selected_tools), []
        query = raw_prompt[7:].strip() if slash_search else raw_prompt
        if not query:
            raise ValueError("search query cannot be empty")
        config = self.web_search
        provider = os.getenv("GOPI_SEARCH_PROVIDER", str(config.get("provider", "searxng"))).lower()
        maximum = int(config.get("max_results", 3))
        timeout = float(config.get("timeout_seconds", 10.0))
        if provider == "searxng":
            results = await asyncio.to_thread(
                search_searxng,
                query,
                max_results=maximum,
                timeout=timeout,
                endpoint=os.getenv("GOPI_SEARXNG_URL", str(config.get("searxng_endpoint", "http://localhost:8080/search"))),
            )
        elif provider == "brave":
            results = await asyncio.to_thread(
                search_brave,
                query,
                os.getenv("GOPI_SEARCH_API_KEY", ""),
                max_results=maximum,
                timeout=timeout,
                endpoint=str(config.get("brave_endpoint", "https://api.search.brave.com/res/v1/web/search")),
            )
        else:
            raise ValueError(f"unsupported search provider: {provider}")
        if not results:
            return None, []
        search_prompt = build_search_prompt(
            query,
            results,
            description_char_limit=int(config.get("description_char_limit", 200)),
        )
        local_context = tool_context(raw_prompt, selected_tools)
        if local_context != raw_prompt:
            search_prompt = f"{search_prompt}\n\nAdditional user and trusted tool context:\n{local_context}"
        return search_prompt, results

    def _format_new_conversation(self, system_prompt: str, user_prompt: str) -> str:
        prompt = format_messages(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            add_generation_prompt=True,
        )
        prompt_ids = self.generator.tokenizer.encode(prompt, add_bos=True, allowed_special="all")
        if len(prompt_ids) < getattr(self.generator, "max_positions", float("inf")):
            return prompt
        logger.warning("System prompt exceeds this model's context; using compact safety prompt")
        return format_messages(
            [{"role": "system", "content": COMPACT_SAFETY_PROMPT}, {"role": "user", "content": user_prompt}],
            add_generation_prompt=True,
        )

    async def _stream_steps(self, prompt: str, options: dict):
        for step in self.generator.stream(prompt, **options):
            yield step
            await asyncio.sleep(0)

def backend_from_environment() -> ConfiguredModelBackend:
    inference_path = Path(os.getenv("GOPI_INFERENCE_CONFIG", "configs/inference.yaml"))
    inference = load_yaml(inference_path) if inference_path.is_file() else {}
    serving = inference.get("serving", {})
    if not isinstance(serving, dict):
        raise ValueError("inference serving configuration must be a mapping")
    return ConfiguredModelBackend(
        model_config=os.getenv("GOPI_MODEL_CONFIG", str(serving.get("model_config", "configs/model.cpu.yaml"))),
        tokenizer_path=os.getenv("GOPI_TOKENIZER_PATH", str(serving.get("tokenizer_path", "data/tokenizer"))),
        checkpoint_path=os.getenv("GOPI_CHECKPOINT_PATH", str(serving.get("checkpoint_path", "checkpoints/latest/model.pt"))),
        device=os.getenv("GOPI_DEVICE", str(serving.get("device", "auto"))),
        session_store_path=os.getenv("GOPI_SESSION_STORE", str(serving.get("session_store_path", "data/cache/sessions.sqlite"))),
        system_prompt=str(inference.get("system_prompt", "You are Gopi, a helpful assistant.")),
        response_format=os.getenv("GOPI_RESPONSE_FORMAT", str(inference.get("response_format", "plain"))),
        context_tokens=int((inference.get("context_memory") or {}).get("max_tokens", 1536)),
        web_search=inference.get("web_search") if isinstance(inference.get("web_search"), dict) else {},
    )
