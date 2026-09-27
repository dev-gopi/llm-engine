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
    "ArtifactRecord",
    "LocalArtifactStorage",
    "S3ArtifactStorage",
    "CapabilitySnapshot",
    "build_capability_snapshot",
    "ProviderContext",
    "ProviderRegistry",
    "RouteDecision",
    "route_multimodal",
    "ArtifactNotFoundError",
    "DependencyMissingError",
    "EnergyVAD",
    "GenerationFailedError",
    "GenerationValidationError",
    "GPUOutOfMemoryError",
    "GPUScheduler",
    "HuggingFaceASRProvider",
    "HuggingFaceTTSProvider",
    "JobCancelledError",
    "JobNotFoundError",
    "MediaValidationError",
    "METRICS",
    "Metrics",
    "ModelNotAvailableError",
    "OmniError",
    "ProviderUnavailableError",
    "ResourceRequest",
    "SpeechConfig",
    "SpeechToSpeechPipeline",
    "UnsupportedCapabilityError",
    "VideoUnderstandingPipeline",
    "CoquiXTTSVoiceCloningProvider",
    "HuggingFaceAudioUnderstandingProvider",
    "HuggingFaceVideoUnderstandingProvider",
    "NativeLatentImageProvider",
]
