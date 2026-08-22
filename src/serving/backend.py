"""Load the trained Gopi model and expose it to the serving runtime."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

from inference.generator import Generator
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
    ) -> None:
        self.model_config = Path(model_config)
        self.tokenizer_path = Path(tokenizer_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.generator: Generator | None = None

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
        load_checkpoint(self.checkpoint_path, model, map_location=device)
        self.generator = Generator(model, tokenizer, device=device)
        logger.info("Loaded model checkpoint %s on %s", self.checkpoint_path, device)

    async def generate(self, request: GenerateRequest) -> BackendGeneration:
        if self.generator is None:
            raise BackendUnavailableError("generation backend is not loaded")
        result = self.generator.generate(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            seed=request.seed,
            stop=request.stop,
        )
        return BackendGeneration(
            text=result.text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=len(result.token_ids),
            finish_reason=FinishReason(result.finish_reason),
        )

    async def stream(self, request: GenerateRequest) -> AsyncIterator[BackendStreamEvent]:
        if self.generator is None:
            raise BackendUnavailableError("generation backend is not loaded")
        result = self.generator.generate(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            seed=request.seed,
            stop=request.stop,
        )
        for index, token_id in enumerate(result.token_ids, 1):
            yield BackendStreamEvent(
                token=self.generator.tokenizer.decode([token_id], skip_special_tokens=True),
                token_id=token_id,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=index,
            )
        yield BackendStreamEvent(
            finish_reason=FinishReason(result.finish_reason),
            prompt_tokens=result.prompt_tokens,
            completion_tokens=len(result.token_ids),
        )

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
    )
