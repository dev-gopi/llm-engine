"""Concrete optional multimodal providers.

Heavy dependencies are loaded lazily so importing the serving stack never pulls
large model frameworks unless an operator explicitly configures a provider.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from .errors import (
    GenerationFailedError,
    MediaValidationError,
    ProviderUnavailableError,
)
from .providers import GenerationArtifact, GenerationResult, ProviderContext


class HuggingFaceAudioUnderstandingProvider:
    id = "hf-audio-understanding"
    capabilities = frozenset({"audio_understanding", "audio_input"})

    def __init__(
        self,
        model: str,
        *,
        task: str = "audio-classification",
        device: str | int = "cpu",
        asr_provider: Any | None = None,
    ) -> None:
        self.model, self.task, self.device, self.asr_provider = (
            model,
            task,
            device,
            asr_provider,
        )
        self._pipe = None
        self._lock = threading.Lock()
        self._error = None

    @classmethod
    def from_env(cls, *, asr_provider: Any | None = None):
        model = os.getenv("GOPI_AUDIO_UNDERSTANDING_MODEL", "").strip()
        if not model:
            return None
        return cls(
            model,
            task=os.getenv("GOPI_AUDIO_UNDERSTANDING_TASK", "audio-classification"),
            device=os.getenv("GOPI_AUDIO_UNDERSTANDING_DEVICE", "cpu"),
            asr_provider=asr_provider,
        )

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        if self._error:
            raise ProviderUnavailableError(self._error)
        with self._lock:
            if self._pipe is not None:
                return self._pipe
            try:
                from transformers import pipeline

                kwargs = {"task": self.task, "model": self.model}
                if self.device != "cpu":
                    kwargs["device"] = self.device
                self._pipe = pipeline(**kwargs)
            except Exception as exc:
                self._error = (
                    f"failed to load audio understanding model {self.model!r}: {exc}"
                )
                raise ProviderUnavailableError(self._error) from exc
        return self._pipe

    def is_available(self) -> bool:
        try:
            self._load()
            return True
        except Exception:
            return False

    def understand_audio(
        self, request: dict[str, Any], context: ProviderContext
    ) -> dict[str, Any]:
        path = Path(str(request.get("path", "")))
        if not path.is_file():
            raise MediaValidationError("audio file does not exist")
        pipe = self._load()
        kwargs = {}
        top_k = int(request.get("top_k", 5))
        question = str(request.get("question") or "").strip()
        if self.task == "audio-classification":
            kwargs["top_k"] = max(1, min(50, top_k))
        try:
            result = pipe(str(path), **kwargs)
        except Exception as exc:
            raise GenerationFailedError(f"audio understanding failed: {exc}") from exc
        transcript = None
        if self.asr_provider is not None and self.asr_provider.is_available():
            try:
                transcript = self.asr_provider.transcribe({"path": str(path)}, context)
            except Exception:
                transcript = None
        return {
            "model": self.model,
            "task": self.task,
            "question": question or None,
            "result": result,
            "transcript": transcript,
        }


class HuggingFaceVideoUnderstandingProvider:
    id = "hf-video-understanding"
    capabilities = frozenset({"video_understanding", "video_input"})

    def __init__(
        self,
        model: str,
        *,
        task: str = "video-classification",
        device: str | int = "cpu",
    ) -> None:
        self.model, self.task, self.device = model, task, device
        self._pipe = None
        self._lock = threading.Lock()
        self._error = None

    @classmethod
    def from_env(cls):
        model = os.getenv("GOPI_VIDEO_UNDERSTANDING_MODEL", "").strip()
        if not model:
            return None
        return cls(
            model,
            task=os.getenv("GOPI_VIDEO_UNDERSTANDING_TASK", "video-classification"),
            device=os.getenv("GOPI_VIDEO_UNDERSTANDING_DEVICE", "cpu"),
        )

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        if self._error:
            raise ProviderUnavailableError(self._error)
        with self._lock:
            if self._pipe is not None:
                return self._pipe
            try:
                from transformers import pipeline

                kwargs = {"task": self.task, "model": self.model}
                if self.device != "cpu":
                    kwargs["device"] = self.device
                self._pipe = pipeline(**kwargs)
            except Exception as exc:
                self._error = (
                    f"failed to load video understanding model {self.model!r}: {exc}"
                )
                raise ProviderUnavailableError(self._error) from exc
        return self._pipe

    def is_available(self) -> bool:
        try:
            self._load()
            return True
        except Exception:
            return False

    def understand_video(
        self, request: dict[str, Any], context: ProviderContext
    ) -> dict[str, Any]:
        path = Path(str(request.get("path", "")))
        if not path.is_file():
            raise MediaValidationError("video file does not exist")
        kwargs = {}
        if self.task == "video-classification":
            kwargs["top_k"] = max(1, min(50, int(request.get("top_k", 5))))
        try:
            result = self._load()(str(path), **kwargs)
        except Exception as exc:
            raise GenerationFailedError(f"video understanding failed: {exc}") from exc
        return {
            "model": self.model,
            "task": self.task,
            "question": request.get("question"),
            "result": result,
        }


class CoquiXTTSVoiceCloningProvider:
    id = "coqui-xtts"
    capabilities = frozenset({"voice_cloning", "text_to_speech", "audio_output"})

    def __init__(
        self,
        model: str,
        *,
        output_dir: str | Path = "outputs/voice_cloning",
        gpu: bool | None = None,
    ) -> None:
        self.model = model
        self.output_dir = Path(output_dir)
        self.gpu = gpu
        self._tts = None
        self._error = None
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls):
        model = os.getenv("GOPI_VOICE_CLONING_MODEL", "").strip()
        if not model:
            return None
        gpu = os.getenv("GOPI_VOICE_CLONING_GPU")
        return cls(
            model,
            output_dir=os.getenv(
                "GOPI_VOICE_CLONING_OUTPUT_DIR", "outputs/voice_cloning"
            ),
            gpu=None if gpu is None else gpu.lower() in {"1", "true", "yes", "on"},
        )

    def _load(self):
        if self._tts is not None:
            return self._tts
        if self._error:
            raise ProviderUnavailableError(self._error)
        with self._lock:
            if self._tts is not None:
                return self._tts
            try:
                from TTS.api import TTS

                self._tts = TTS(model_name=self.model, progress_bar=False, gpu=self.gpu)
            except Exception as exc:
                self._error = f"failed to load Coqui XTTS model {self.model!r}: {exc}"
                raise ProviderUnavailableError(self._error) from exc
        return self._tts

    def is_available(self) -> bool:
        try:
            self._load()
            return True
        except Exception:
            return False

    def clone_voice(
        self, request: dict[str, Any], context: ProviderContext
    ) -> GenerationResult:
        text = str(request.get("text", "")).strip()
        ref = Path(str(request.get("reference_audio", request.get("path", ""))))
        if not text:
            raise MediaValidationError("text is required")
        if not ref.is_file():
            raise MediaValidationError("reference_audio does not exist")
        language = str(
            request.get("language") or os.getenv("GOPI_VOICE_CLONING_LANGUAGE", "en")
        )
        output = Path(
            request.get("output") or self.output_dir / f"{context.request_id}.wav"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._load().tts_to_file(
                text=text,
                speaker_wav=str(ref),
                language=language,
                file_path=str(output),
            )
        except Exception as exc:
            raise GenerationFailedError(f"voice cloning failed: {exc}") from exc
        return GenerationResult(
            [GenerationArtifact(output, "audio/wav", {"language": language})],
            metadata={"provider": self.id, "model": self.model},
        )


class NativeLatentImageProvider:
    """Checkpoint-backed native latent diffusion text-to-image provider.

    Expected checkpoints are explicit and independent: U-Net, VAE and text encoder.
    This avoids silently treating an architecture-only profile as a pretrained model.
    """

    id = "native-latent-image"
    capabilities = frozenset({"image_generation", "image_editing", "image_output"})

    def __init__(
        self,
        config_path: str | Path,
        model_checkpoint: str | Path,
        vae_checkpoint: str | Path,
        text_encoder_checkpoint: str | Path,
        tokenizer_path: str | Path,
        *,
        output_dir: str | Path = "outputs/images",
        device: str | None = None,
    ):
        self.config_path = Path(config_path)
        self.model_checkpoint = Path(model_checkpoint)
        self.vae_checkpoint = Path(vae_checkpoint)
        self.text_encoder_checkpoint = Path(text_encoder_checkpoint)
        self.tokenizer_path = Path(tokenizer_path)
        self.output_dir = Path(output_dir)
        self.device = device
        self._runtime = None
        self._lock = threading.Lock()
        self._error = None

    @classmethod
    def from_env(cls):
        vals = [
            os.getenv("GOPI_LATENT_IMAGE_CONFIG"),
            os.getenv("GOPI_LATENT_IMAGE_CHECKPOINT"),
            os.getenv("GOPI_LATENT_VAE_CHECKPOINT"),
            os.getenv("GOPI_LATENT_TEXT_ENCODER_CHECKPOINT"),
            os.getenv("GOPI_LATENT_IMAGE_TOKENIZER"),
        ]
        if not all(vals):
            return None
        return cls(
            *vals,
            output_dir=os.getenv("GOPI_LATENT_IMAGE_OUTPUT_DIR", "outputs/images"),
            device=os.getenv("GOPI_LATENT_IMAGE_DEVICE"),
        )

    def _load(self):
        if self._runtime is not None:
            return self._runtime
        if self._error:
            raise ProviderUnavailableError(self._error)
        with self._lock:
            if self._runtime is not None:
                return self._runtime
            try:
                import torch

                from diffusion.latent_pipeline import LatentDiffusionPipeline
                from diffusion.scheduler import DiffusionScheduler
                from diffusion.text_encoder import DiffusionTextEncoder
                from diffusion.unet import SmallUNet
                from diffusion.vae import AutoencoderKL
                from tokenizer.encoder import Tokenizer
                from training.checkpoint import load_checkpoint
                from utils.config import load_yaml

                cfg = load_yaml(self.config_path)
                tok = Tokenizer.load(self.tokenizer_path)
                dev = torch.device(
                    self.device or ("cuda" if torch.cuda.is_available() else "cpu")
                )
                vae = AutoencoderKL.from_config(cfg).to(dev)
                load_checkpoint(
                    self.vae_checkpoint,
                    vae,
                    map_location=dev,
                    use_ema=True,
                    restore_rng=False,
                )
                model_cfg = dict(cfg)
                model_cfg["image_channels"] = int(cfg.get("latent_channels", 4))
                model = SmallUNet.from_config(model_cfg).to(dev)
                load_checkpoint(
                    self.model_checkpoint,
                    model,
                    map_location=dev,
                    use_ema=True,
                    restore_rng=False,
                )
                text = DiffusionTextEncoder.from_config(
                    cfg, vocab_size=tok.vocab_size
                ).to(dev)
                load_checkpoint(
                    self.text_encoder_checkpoint,
                    text,
                    map_location=dev,
                    use_ema=True,
                    restore_rng=False,
                )
                sched = DiffusionScheduler(
                    int(cfg.get("timesteps", 1000)),
                    float(cfg.get("beta_start", 1e-4)),
                    float(cfg.get("beta_end", 2e-2)),
                    device=dev,
                    schedule=str(cfg.get("noise_schedule", "cosine")),
                )
                for m in (vae, model, text):
                    m.eval()
                self._runtime = (
                    cfg,
                    tok,
                    dev,
                    LatentDiffusionPipeline(
                        vae,
                        model,
                        sched,
                        text,
                        latent_scale=float(cfg.get("latent_scale", 0.18215)),
                    ),
                )
            except Exception as exc:
                self._error = f"failed to load native latent image runtime: {exc}"
                raise ProviderUnavailableError(self._error) from exc
        return self._runtime

    def is_available(self) -> bool:
        try:
            self._load()
            return True
        except Exception:
            return False

    def generate_image(
        self, request: dict[str, Any], context: ProviderContext
    ) -> GenerationResult:
        import torch

        from image_data.processor import tensor_to_image

        cfg, tok, dev, pipe = self._load()
        prompt = str(request.get("prompt", "")).strip()
        if not prompt:
            raise MediaValidationError("prompt is required")
        n = max(1, min(10, int(request.get("n", 1))))
        size = str(
            request.get("size")
            or f"{cfg.get('image_size', 256)}x{cfg.get('image_size', 256)}"
        )
        width, height = [int(x) for x in size.lower().split("x", 1)]
        if width != height:
            raise MediaValidationError(
                "native latent runtime currently requires square output"
            )
        max_len = int(cfg.get("text_max_length", 128))
        pad = int(cfg.get("text_padding_idx", 0))
        ids = tok.encode(prompt)[:max_len]
        ids = ids or [pad]
        mask = [1] * len(ids)
        ids += [pad] * (max_len - len(ids))
        mask += [0] * (max_len - len(mask))
        token_ids = torch.tensor([ids] * n, device=dev, dtype=torch.long)
        attention = torch.tensor([mask] * n, device=dev, dtype=torch.bool)
        seed = int(request.get("seed", 42))
        gen = torch.Generator(device=dev).manual_seed(seed)
        images = pipe.sample(
            n,
            width,
            device=dev,
            token_ids=token_ids,
            attention_mask=attention,
            guidance_scale=float(
                request.get("guidance_scale", cfg.get("guidance_scale", 5))
            ),
            inference_steps=int(request.get("steps", cfg.get("inference_steps", 50))),
            generator=gen,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        artifacts = []
        for i, image in enumerate(images):
            path = self.output_dir / f"{context.request_id}-{i:02d}.png"
            tensor_to_image(image).save(path)
            artifacts.append(
                GenerationArtifact(path, "image/png", {"seed": seed, "index": i})
            )
        return GenerationResult(
            artifacts, metadata={"provider": self.id, "config": str(self.config_path)}
        )

    def edit_image(
        self, request: dict[str, Any], context: ProviderContext
    ) -> GenerationResult:
        import torch

        from image_data.processor import load_image, tensor_to_image

        cfg, tok, dev, pipe = self._load()
        prompt = str(request.get("prompt", "")).strip()
        source = Path(str(request.get("image") or request.get("path") or ""))
        if not source.is_file():
            raise MediaValidationError("image file does not exist")
        if not prompt:
            raise MediaValidationError("prompt is required")
        size = int(request.get("image_size") or cfg.get("image_size", 256))
        if size % pipe.vae.downsample_factor:
            raise MediaValidationError(
                "image size must be divisible by the VAE downsample factor"
            )
        strength = float(request.get("strength", 0.5))
        if not 0.0 <= strength <= 1.0:
            raise MediaValidationError("strength must be between zero and one")
        n = max(1, min(10, int(request.get("n", 1))))
        max_len = int(cfg.get("text_max_length", 128))
        pad = int(cfg.get("text_padding_idx", 0))
        ids = tok.encode(prompt)[:max_len]
        ids = ids or [pad]
        mask = [1] * len(ids)
        ids += [pad] * (max_len - len(ids))
        mask += [0] * (max_len - len(mask))
        token_ids = torch.tensor([ids] * n, device=dev, dtype=torch.long)
        attention = torch.tensor([mask] * n, device=dev, dtype=torch.bool)
        image = (
            load_image(source, size)
            .unsqueeze(0)
            .to(dev)
            .expand(n, -1, -1, -1)
            .contiguous()
        )
        seed = int(request.get("seed", 42))
        gen = torch.Generator(device=dev).manual_seed(seed)
        try:
            images = pipe.edit(
                image,
                device=dev,
                strength=strength,
                token_ids=token_ids,
                attention_mask=attention,
                guidance_scale=float(
                    request.get("guidance_scale", cfg.get("guidance_scale", 5))
                ),
                inference_steps=int(
                    request.get("steps", cfg.get("inference_steps", 50))
                ),
                generator=gen,
            )
        except Exception as exc:
            raise GenerationFailedError(
                f"native latent image editing failed: {exc}"
            ) from exc
        self.output_dir.mkdir(parents=True, exist_ok=True)
        artifacts = []
        for i, out in enumerate(images):
            path = self.output_dir / f"{context.request_id}-edit-{i:02d}.png"
            tensor_to_image(out).save(path)
            artifacts.append(
                GenerationArtifact(
                    path,
                    "image/png",
                    {
                        "seed": seed,
                        "index": i,
                        "strength": strength,
                        "source": str(source),
                    },
                )
            )
        return GenerationResult(
            artifacts,
            metadata={
                "provider": self.id,
                "config": str(self.config_path),
                "operation": "edit",
            },
        )


__all__ = [
    "CoquiXTTSVoiceCloningProvider",
    "HuggingFaceAudioUnderstandingProvider",
    "HuggingFaceVideoUnderstandingProvider",
    "NativeLatentImageProvider",
]
