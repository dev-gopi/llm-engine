"""Map visual features into the language model embedding space."""

from torch import Tensor, nn


class VisionProjector(nn.Module):
    def __init__(self, vision_size: int, language_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        if vision_size <= 0 or language_size <= 0:
            raise ValueError("vision_size and language_size must be positive")
        self.network = nn.Sequential(
            nn.Linear(vision_size, language_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(language_size, language_size),
            nn.LayerNorm(language_size),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError("visual features must have shape [batch, tokens, hidden]")
        return self.network(features)
