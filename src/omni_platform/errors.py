"""Typed errors shared by multimodal providers, jobs, storage, and APIs."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class OmniError(Exception):
    message: str
    code: str = "omni_error"
    http_status: int = 500
    retryable: bool = False

    def __str__(self) -> str:
        return self.message


class ModelNotAvailableError(OmniError):
    def __init__(self, message: str = "model is not available") -> None:
        super().__init__(message, "model_not_available", 503, True)


class UnsupportedCapabilityError(OmniError):
    def __init__(self, message: str = "capability is not supported") -> None:
        super().__init__(message, "unsupported_capability", 400, False)


class GenerationValidationError(OmniError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "generation_validation_error", 422, False)


class GenerationFailedError(OmniError):
    def __init__(self, message: str = "generation failed", *, retryable: bool = True) -> None:
        super().__init__(message, "generation_failed", 500, retryable)


class GPUOutOfMemoryError(OmniError):
    def __init__(self, message: str = "GPU memory exhausted") -> None:
        super().__init__(message, "gpu_out_of_memory", 503, True)


class ArtifactNotFoundError(OmniError):
    def __init__(self, message: str = "artifact not found") -> None:
        super().__init__(message, "artifact_not_found", 404, False)


class JobNotFoundError(OmniError):
    def __init__(self, message: str = "job not found") -> None:
        super().__init__(message, "job_not_found", 404, False)


class JobCancelledError(OmniError):
    def __init__(self, message: str = "job cancelled") -> None:
        super().__init__(message, "job_cancelled", 409, False)


class MediaValidationError(OmniError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "media_validation_error", 422, False)


class ProviderUnavailableError(OmniError):
    def __init__(self, message: str = "provider unavailable") -> None:
        super().__init__(message, "provider_unavailable", 503, True)


class DependencyMissingError(OmniError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "dependency_missing", 503, False)
