"""Small, dependency-light vision models."""

from .encoder import VisionEncoder
from .classifier import VisionClassifier

__all__ = ["VisionClassifier", "VisionEncoder"]
