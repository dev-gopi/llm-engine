"""Optional native multimodal serving support.

Vision is capability-gated: this code is only activated when a trained
VisionLanguageModel checkpoint is explicitly configured and successfully
loaded. Text-only checkpoints never advertise or silently simulate vision.
"""
from __future__ import annotations

import base64
import binascii
import io
import ipaddress
import socket
from typing import Any
from urllib.parse import urlparse

import httpx
import numpy as np
import torch

from inference.generator import GenerationResult, Generator
from inference.sampler import JSONSchemaConstraint
from multimodal.model import VisionLanguageModel


def message_image_urls(messages: list[dict[str, Any]] | None) -> list[str]:
    urls: list[str] = []
    for message in messages or []:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                urls.append(image_url["url"])
    return urls


def has_image_input(messages: list[dict[str, Any]] | None) -> bool:
    return bool(message_image_urls(messages))


def _decode_data_url(url: str, *, max_bytes: int) -> bytes:
    header, separator, payload = url.partition(",")
    if not separator or not header.lower().startswith("data:image/"):
        raise ValueError("vision input must use an image data URL")
    if ";base64" not in header.lower():
        raise ValueError("image data URLs must be base64 encoded")
    try:
        value = base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("image data URL contains invalid base64") from exc
    if not value:
        raise ValueError("image data URL is empty")
    if len(value) > max_bytes:
        raise ValueError(f"image exceeds the {max_bytes}-byte serving limit")
    return value


def _host_is_public(hostname: str) -> bool:
    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
        }
    except OSError:
        return False
    if not addresses:
        return False
    for raw in addresses:
        try:
            address = ipaddress.ip_address(raw)
        except ValueError:
            return False
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
        ):
            return False
    return True


async def _download_remote_image(url: str, *, max_bytes: int, timeout_seconds: float) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("remote image URL must use http or https")
    if not await asyncio_to_thread(_host_is_public, parsed.hostname):
        raise ValueError("remote image URL must resolve only to public addresses")
    # Redirects are disabled so an approved public host cannot redirect into a
    # private network. Content length is checked before and during streaming.
    async with httpx.AsyncClient(follow_redirects=False, timeout=timeout_seconds) as client:
        async with client.stream("GET", url, headers={"Accept": "image/*"}) as response:
            response.raise_for_status()
            declared = response.headers.get("content-length")
            if declared and int(declared) > max_bytes:
                raise ValueError(f"image exceeds the {max_bytes}-byte serving limit")
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"image exceeds the {max_bytes}-byte serving limit")
                chunks.append(chunk)
            return b"".join(chunks)


async def asyncio_to_thread(function, *args):
    """Tiny local wrapper to keep this module import-light and testable."""
    import asyncio

    return await asyncio.to_thread(function, *args)


async def image_url_to_bytes(
    url: str,
    *,
    allow_remote: bool,
    max_bytes: int,
    timeout_seconds: float = 10.0,
) -> bytes:
    if url.startswith("data:"):
        return _decode_data_url(url, max_bytes=max_bytes)
    if not allow_remote:
        raise ValueError(
            "native vision accepts base64 image data URLs; set GOPI_VISION_ALLOW_REMOTE_IMAGES=true "
            "to permit public http/https image URLs"
        )
    return await _download_remote_image(url, max_bytes=max_bytes, timeout_seconds=timeout_seconds)


def image_bytes_to_tensor(
    payload: bytes,
    *,
    image_size: int,
    normalization: str = "zero_one",
) -> torch.Tensor:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("native vision requires: pip install -e '.[images]'") from exc
    if normalization not in {"zero_one", "minus_one_one", "imagenet"}:
        raise ValueError("vision normalization must be zero_one, minus_one_one, or imagenet")
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            # This intentionally matches ImageTextSFTDataset: fixed square
            # resize followed by float RGB conversion.
            image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
            array = np.asarray(image, dtype=np.float32).copy()
    except Exception as exc:
        raise ValueError("image payload could not be decoded") from exc
    tensor = torch.from_numpy(array).permute(2, 0, 1).div(255.0)
    if normalization == "minus_one_one":
        tensor = tensor.mul(2).sub(1)
    elif normalization == "imagenet":
        mean = tensor.new_tensor((0.485, 0.456, 0.406))[:, None, None]
        std = tensor.new_tensor((0.229, 0.224, 0.225))[:, None, None]
        tensor = (tensor - mean) / std
    return tensor


async def load_image_tensors(
    urls: list[str],
    *,
    image_size: int,
    normalization: str,
    allow_remote: bool,
    max_bytes: int,
    max_images: int,
) -> torch.Tensor:
    if not urls:
        raise ValueError("at least one image is required")
    if len(urls) > max_images:
        raise ValueError(f"native vision accepts at most {max_images} images per request")
    tensors: list[torch.Tensor] = []
    for url in urls:
        payload = await image_url_to_bytes(
            url,
            allow_remote=allow_remote,
            max_bytes=max_bytes,
        )
        tensors.append(
            image_bytes_to_tensor(
                payload,
                image_size=image_size,
                normalization=normalization,
            )
        )
    return torch.stack(tensors)


class MultimodalGenerator:
    """Correctness-first autoregressive decoder over prompt + visual embeddings.

    The existing text Generator uses a KV cache. VisionLanguageModel currently
    exposes a full-embedding forward path but no multimodal KV-cache prefill, so
    this implementation recomputes the short sequence each step. It is slower,
    but it makes the trained projector usable without changing checkpoint
    layouts. A future optimized backend can replace it without changing the API.
    """

    def __init__(self, model: VisionLanguageModel, text_generator: Generator) -> None:
        self.model = model.eval()
        self.text_generator = text_generator
        self.tokenizer = text_generator.tokenizer
        self.device = text_generator.device
        self.max_positions = int(model.language_model.max_positions)
        self.eos_token_id = text_generator.eos_token_id

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        images: torch.Tensor,
        *,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        min_p: float = 0.0,
        repetition_penalty: float = 1.1,
        no_repeat_ngram_size: int = 3,
        min_tokens: int = 1,
        seed: int | None = None,
        stop: list[str] | None = None,
        allow_special_tokens: bool = True,
        json_schema: dict[str, Any] | None = None,
    ) -> GenerationResult:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [images, 3, height, width]")
        prompt_ids = self.tokenizer.encode(
            prompt,
            add_bos=True,
            allowed_special="all" if allow_special_tokens else (),
        )
        if not prompt_ids:
            raise ValueError("prompt encoded to no tokens")
        visual_count = int(images.shape[0]) * int(self.model.visual_tokens)
        used = len(prompt_ids) + visual_count
        if used >= self.max_positions:
            raise ValueError(
                f"text plus visual prompt uses {used} tokens but model context is {self.max_positions}"
            )
        limit = min(max_tokens, self.max_positions - used)
        if limit < 1:
            raise ValueError("model context has no room for a completion")

        language = self.model.language_model
        parameter = next(self.model.vision_encoder.parameters())
        images = images.to(device=self.device, dtype=parameter.dtype)
        visual = self.model.encode_images(images).reshape(1, -1, language.dim)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        embeddings = torch.cat((language.tok(prompt_tensor), visual), dim=1)

        random = torch.Generator(device=self.device)
        if seed is not None:
            random.manual_seed(seed)
        generated: list[int] = []
        all_ids = list(prompt_ids)
        stop_sequences = stop or []
        finish_reason = "length"
        constraint = JSONSchemaConstraint(json_schema) if json_schema is not None else None

        for step in range(limit):
            logits = self.model._language_forward_from_embeddings(embeddings)
            next_logits = logits[:, -1, :].clone()
            self.text_generator._apply_repetition_penalty(
                next_logits, set(all_ids), repetition_penalty
            )
            self.text_generator._apply_no_repeat_ngram(
                next_logits, all_ids, no_repeat_ngram_size
            )
            self.text_generator._suppress_special_tokens(
                next_logits, len(generated), min_tokens
            )
            if constraint is not None:
                next_logits = constraint.filter_logits(
                    next_logits, generated, self.tokenizer, candidate_k=256
                )
            next_id = int(self.text_generator.sampler(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                generator=random,
            ).item())
            if self.eos_token_id is not None and next_id == self.eos_token_id:
                finish_reason = "stop"
                break
            generated.append(next_id)
            all_ids.append(next_id)
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            if any(sequence in text for sequence in stop_sequences):
                finish_reason = "stop"
                text = self.text_generator._trim_stop(text, stop_sequences)
                if constraint is not None and not constraint.validate(text):
                    raise ValueError("generated text failed the requested output constraint")
                return GenerationResult(text, tuple(generated), used, finish_reason)
            if step + 1 < limit:
                token = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                embeddings = torch.cat((embeddings, language.tok(token)), dim=1)

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        text = self.text_generator._trim_repeated_text(text)
        if constraint is not None and not constraint.validate(text):
            raise ValueError("generated text failed the requested output constraint")
        return GenerationResult(text, tuple(generated), used, finish_reason)
