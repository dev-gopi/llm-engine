"""Non-invasive vision-language wrapper around MiniGPT."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from model.gpt import MiniGPT
from vision.encoder import VisionEncoder
from .projector import VisionProjector


class VisionLanguageModel(nn.Module):
    """Connect visual tokens to a text model without changing its checkpoint layout."""

    def __init__(
        self,
        vision_encoder: VisionEncoder,
        language_model: MiniGPT,
        *,
        visual_tokens: int = 16,
        projector_dropout: float = 0.0,
        freeze_vision: bool = True,
        freeze_language: bool = True,
    ) -> None:
        super().__init__()
        if visual_tokens <= 0 or visual_tokens > vision_encoder.patch_embedding.num_patches:
            raise ValueError("visual_tokens must fit within the available image patches")
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.visual_tokens = visual_tokens
        self.freeze_vision = freeze_vision
        self.freeze_language = freeze_language
        self.projector = VisionProjector(
            vision_encoder.hidden_size, language_model.dim, projector_dropout
        )
        self._set_trainable(vision_encoder, not freeze_vision)
        self._set_trainable(language_model, not freeze_language)

    def train(self, mode: bool = True) -> "VisionLanguageModel":
        super().train(mode)
        # Frozen backbones must also keep dropout disabled while the projector
        # trains, otherwise identical images/prompts produce moving targets.
        if self.freeze_vision:
            self.vision_encoder.eval()
        if self.freeze_language:
            self.language_model.eval()
        return self

    @staticmethod
    def _set_trainable(module: nn.Module, trainable: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = trainable

    def encode_images(self, images: Tensor) -> Tensor:
        features = self.vision_encoder(images)[:, 1 : self.visual_tokens + 1]
        return self.projector(features)

    def build_input_embeddings(
        self, images: Tensor, prompt_ids: Tensor, response_ids: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if prompt_ids.ndim != 2 or (response_ids is not None and response_ids.ndim != 2):
            raise ValueError("prompt_ids and response_ids must have shape [batch, sequence]")
        if images.shape[0] != prompt_ids.shape[0]:
            raise ValueError("image and prompt batch sizes must match")
        visual = self.encode_images(images)
        prompt = self.language_model.tok(prompt_ids)
        pieces = [prompt, visual]
        loss_mask = [torch.zeros(prompt.shape[:2], dtype=torch.bool, device=prompt.device)]
        loss_mask.append(torch.zeros(visual.shape[:2], dtype=torch.bool, device=visual.device))
        if response_ids is not None:
            if response_ids.shape[0] != images.shape[0]:
                raise ValueError("image and response batch sizes must match")
            response = self.language_model.tok(response_ids)
            pieces.append(response)
            loss_mask.append(torch.ones(response.shape[:2], dtype=torch.bool, device=response.device))
        embeddings = torch.cat(pieces, dim=1)
        if embeddings.shape[1] > self.language_model.max_positions:
            raise ValueError("combined visual and text sequence exceeds model context length")
        return embeddings, torch.cat(loss_mask, dim=1)

    def forward(
        self, images: Tensor, prompt_ids: Tensor, response_ids: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        embeddings, loss_mask = self.build_input_embeddings(images, prompt_ids, response_ids)
        logits = self._language_forward_from_embeddings(embeddings)
        return logits, loss_mask

    def _language_forward_from_embeddings(self, hidden_states: Tensor) -> Tensor:
        """Use existing public submodules while leaving MiniGPT source untouched."""
        if self.language_model.position_type == "learned" and self.language_model.pos is not None:
            batch, length = hidden_states.shape[:2]
            dummy_ids = torch.zeros((batch, length), dtype=torch.long, device=hidden_states.device)
            hidden_states = hidden_states + self.language_model.pos(dummy_ids)
        elif self.language_model.position_type == "sinusoidal" and self.language_model.pos is not None:
            batch, length = hidden_states.shape[:2]
            dummy_ids = torch.zeros((batch, length), dtype=torch.long, device=hidden_states.device)
            hidden_states = hidden_states + self.language_model.pos(dummy_ids)
        hidden_states = self.language_model.embedding_dropout(hidden_states)
        rotary = None
        if self.language_model.rotary_emb is not None:
            rotary = self.language_model.rotary_emb(hidden_states, seq_len=hidden_states.shape[1])
        for block in self.language_model.blocks:
            hidden_states = block(hidden_states, rotary_pos_emb=rotary)
        logits = self.language_model.head(self.language_model.norm(hidden_states))
        if self.language_model.logit_softcap is not None:
            logits = self.language_model.logit_softcap * torch.tanh(
                logits / self.language_model.logit_softcap
            )
        return logits
