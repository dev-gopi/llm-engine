"""OpenAI-compatible embeddings request/response models and service adapter."""
from __future__ import annotations

from typing import Any, Literal

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
        if not values or any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("input must be a non-empty string or list of non-empty strings")
        return values


class EmbeddingItem(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: list[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class EmbeddingsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingItem]
    model: str
    usage: EmbeddingUsage


def create_embeddings(request: EmbeddingsRequest, service: EmbeddingService) -> EmbeddingsResponse:
    result = service.encode(request.texts())
    vectors = result.embeddings
    if request.dimensions is not None and request.dimensions != service.dimension:
        raise ValueError(
            f"embedding model {service.dimension}-dimensional vectors; requested dimensions={request.dimensions} is unsupported"
        )
    if request.encoding_format == "base64":
        # The API contract intentionally rejects base64 until a binary wire
        # format is implemented rather than silently returning a wrong type.
        raise ValueError("encoding_format=base64 is not supported by this embedding backend")
    return EmbeddingsResponse(
        data=[EmbeddingItem(embedding=vector, index=index) for index, vector in enumerate(vectors)],
        model=request.model,
        usage=EmbeddingUsage(prompt_tokens=result.prompt_tokens, total_tokens=result.prompt_tokens),
    )
