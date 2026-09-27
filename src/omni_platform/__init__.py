"""Unified multimodal platform primitives for production serving."""

from .artifacts import ArtifactRecord, LocalArtifactStorage, S3ArtifactStorage
from .capabilities import CapabilitySnapshot, build_capability_snapshot
from .errors import (
    ArtifactNotFoundError,
    DependencyMissingError,
    GenerationFailedError,
    GenerationValidationError,
    GPUOutOfMemoryError,
    JobCancelledError,
    JobNotFoundError,
    MediaValidationError,
    ModelNotAvailableError,
    OmniError,
    ProviderUnavailableError,
    UnsupportedCapabilityError,
)
from .native_multimodal import (
    CoquiXTTSVoiceCloningProvider,
    HuggingFaceAudioUnderstandingProvider,
    HuggingFaceVideoUnderstandingProvider,
    NativeLatentImageProvider,
)
from .observability import METRICS, Metrics
from .providers import ProviderContext, ProviderRegistry
from .router import RouteDecision, route_multimodal
from .scheduler import GPUScheduler, ResourceRequest
from .speech import (
    EnergyVAD,
    HuggingFaceASRProvider,
    HuggingFaceTTSProvider,
    SpeechConfig,
    SpeechToSpeechPipeline,
)
from .video_understanding import VideoUnderstandingPipeline

__all__ = [
    "METRICS",
    "ArtifactNotFoundError",
    "ArtifactRecord",
    "CapabilitySnapshot",
    "CoquiXTTSVoiceCloningProvider",
    "DependencyMissingError",
    "EnergyVAD",
    "GPUOutOfMemoryError",
    "GPUScheduler",
    "GenerationFailedError",
    "GenerationValidationError",
    "HuggingFaceASRProvider",
    "HuggingFaceAudioUnderstandingProvider",
    "HuggingFaceTTSProvider",
    "HuggingFaceVideoUnderstandingProvider",
    "JobCancelledError",
    "JobNotFoundError",
    "LocalArtifactStorage",
    "MediaValidationError",
    "Metrics",
    "ModelNotAvailableError",
    "NativeLatentImageProvider",
    "OmniError",
    "ProviderContext",
    "ProviderRegistry",
    "ProviderUnavailableError",
    "ResourceRequest",
    "RouteDecision",
    "S3ArtifactStorage",
    "SpeechConfig",
    "SpeechToSpeechPipeline",
    "UnsupportedCapabilityError",
    "VideoUnderstandingPipeline",
    "build_capability_snapshot",
    "route_multimodal",
]
