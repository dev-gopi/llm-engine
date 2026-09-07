"""Local, self-contained MiniGPT inference bundles; no training state required."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

import torch
from safetensors.torch import load_model, save_model

from datasets.preprocessor import format_messages
from inference.generator import Generator
from model.config import normalize_model_config
from model.gpt import MiniGPT
from model.vocabulary import adapt_config_to_tokenizer, checkpoint_tokenizer_options
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint


class MiniGPTBackend:
    """Load existing checkpoints or portable bundles and expose generation APIs."""

    def __init__(self, model, tokenizer, model_config, *, device="auto", generation_config=None):
        self.model_config = normalize_model_config(model_config)
        self.generation_config = dict(generation_config or {})
        self.generator = Generator(model, tokenizer, device=device)
        self.model = self.generator.model
        self.tokenizer = tokenizer

    @classmethod
    def from_checkpoint(cls, checkpoint, *, model_config, tokenizer, device="auto",
                        use_ema=True, generation_config=None):
        if not isinstance(tokenizer, Tokenizer):
            tokenizer = Tokenizer.load(tokenizer)
        config = adapt_config_to_tokenizer(normalize_model_config(model_config), tokenizer)
        # Optimizer state in training checkpoints never enters GPU memory.
        model = MiniGPT.from_config(config, device="cpu")
        load_checkpoint(checkpoint, model, map_location="cpu", restore_rng=False,
                        use_ema=use_ema, **checkpoint_tokenizer_options(tokenizer))
        return cls(model, tokenizer, config, device=device, generation_config=generation_config)

    def generate(self, prompt, **options):
        return self.generator.generate(prompt, **(self.generation_config | options))

    def stream(self, prompt, **options):
        return self.generator.stream(prompt, **(self.generation_config | options))

    def generate_batch(self, prompts, **options):
        return self.generator.generate_batch(prompts, **(self.generation_config | options))

    def chat(self, messages, **options):
        prompt = format_messages(messages, add_generation_prompt=True)
        return self.generate(prompt, **(options | {"allow_special_tokens": True}))

    def open_session(self, path, session_id, *, system_prompt="", ttl_seconds=86400):
        """Opt into persistent history at an explicitly selected SQLite path."""
        from inference.chat_session import ChatSession
        return ChatSession(self, path, session_id, system_prompt=system_prompt,
                           ttl_seconds=ttl_seconds)

    def save_pretrained(self, directory):
        """Publish a new local bundle; refuse to replace an existing directory."""
        destination = Path(directory)
        if destination.exists():
            raise FileExistsError(f"bundle already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        try:
            config = {"format_version": 1, "architecture": "MiniGPT",
                      "model_config": self.model_config,
                      "tokenizer_fingerprint": self.tokenizer.fingerprint,
                      "generation_config": self.generation_config}
            (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            self.tokenizer.save(staging / "tokenizer")
            # save_model understands tied embedding/head storage.
            save_model(self.model, str(staging / "model.safetensors"))
            staging.rename(destination)
        except BaseException:
            shutil.rmtree(staging)
            raise
        return destination

    @classmethod
    def from_pretrained(cls, directory, *, device="auto"):
        """Load a local MiniGPT bundle, strictly checking its format and tokenizer."""
        source = Path(directory)
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        if config.get("format_version") != 1 or config.get("architecture") != "MiniGPT":
            raise ValueError("unsupported MiniGPT bundle format")
        tokenizer = Tokenizer.load(source / "tokenizer")
        if tokenizer.fingerprint != config.get("tokenizer_fingerprint"):
            raise ValueError("bundle tokenizer fingerprint mismatch")
        model_config = config["model_config"]
        if tokenizer.vocab_size != int(model_config["vocab_size"]):
            raise ValueError("bundle tokenizer vocabulary mismatch")
        model = MiniGPT.from_config(model_config, device="cpu")
        load_model(model, str(source / "model.safetensors"), strict=True, device="cpu")
        return cls(model, tokenizer, model_config, device=device,
                   generation_config=config.get("generation_config", {}))
