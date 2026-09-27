"""OpenAI-compatible embeddings request/response models and service adapter."""

from __future__ import annotations

import base64
import struct
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from embeddings import EmbeddingService


class EmbeddingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1, max_length=128)
    input: str | list[str]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = Field(default=None, ge=1)

    def texts(self) -> list[str]:
        values = [self.input] if isinstance(self.input, str) else self.input
        if not values or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            raise ValueError(
                "input must be a non-empty string or list of non-empty strings"
            )
        return values


class EmbeddingItem(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: list[float] | str
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class EmbeddingsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingItem]
    model: str
    usage: EmbeddingUsage


def _base64_float32(vector: list[float]) -> str:
    payload = struct.pack(f"<{len(vector)}f", *vector)
    return base64.b64encode(payload).decode("ascii")


def create_embeddings(
    request: EmbeddingsRequest, service: EmbeddingService
) -> EmbeddingsResponse:
    result = service.encode(request.texts(), dimensions=request.dimensions)
    vectors = result.embeddings
    encoded: list[list[float] | str]
    if request.encoding_format == "base64":
        encoded = [_base64_float32(vector) for vector in vectors]
    else:
        encoded = vectors
    return EmbeddingsResponse(
        data=[
            EmbeddingItem(embedding=vector, index=index)
            for index, vector in enumerate(encoded)
        ],
        model=request.model,
        usage=EmbeddingUsage(
            prompt_tokens=result.prompt_tokens, total_tokens=result.prompt_tokens
        ),
    )
