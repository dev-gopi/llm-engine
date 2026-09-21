"""Non-invasive vision-language wrapper around MiniGPT."""

from __future__ import annotations

from dataclasses import dataclass

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
        if freeze_vision:
            self.vision_encoder.eval()
        if freeze_language:
            self.language_model.eval()

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


@dataclass(frozen=True)
class ImageTextSFTExample:
    """One governed image-to-text SFT example."""

    image: Tensor
    prompt_ids: Tensor
    response_ids: Tensor
    sample_id: str = ""


def collate_image_text_sft(examples: list[ImageTextSFTExample], *, pad_token_id: int = 0) -> dict[str, Tensor]:
    """Pad image-text SFT examples and build an assistant-only loss mask."""
    if not examples:
        raise ValueError("examples must not be empty")
    images = torch.stack([example.image for example in examples])
    prompt_width = max(int(example.prompt_ids.numel()) for example in examples)
    response_width = max(int(example.response_ids.numel()) for example in examples)
    prompt = torch.full((len(examples), prompt_width), pad_token_id, dtype=torch.long)
    response = torch.full((len(examples), response_width), pad_token_id, dtype=torch.long)
    prompt_mask = torch.zeros_like(prompt, dtype=torch.bool)
    response_mask = torch.zeros_like(response, dtype=torch.bool)
    for row, example in enumerate(examples):
        pids = example.prompt_ids.reshape(-1).long()
        rids = example.response_ids.reshape(-1).long()
        prompt[row, : pids.numel()] = pids
        response[row, : rids.numel()] = rids
        prompt_mask[row, : pids.numel()] = True
        response_mask[row, : rids.numel()] = True
    return {
        "images": images,
        "prompt_ids": prompt,
        "response_ids": response,
        "prompt_attention_mask": prompt_mask,
        "response_loss_mask": response_mask,
        "sample_ids": [example.sample_id for example in examples],
    }


def multimodal_sft_metrics(logits: Tensor, response_ids: Tensor, response_loss_mask: Tensor) -> dict[str, float]:
    """Compute token accuracy/perplexity for the response portion only."""
    if logits.ndim != 3 or response_ids.ndim != 2 or response_loss_mask.shape != response_ids.shape:
        raise ValueError("invalid multimodal SFT evaluation tensors")
    # logits contain prompt + visual + response positions; compare response token
    # t against the preceding position, excluding the first response token.
    response_start = logits.shape[1] - response_ids.shape[1]
    response_logits = logits[:, response_start - 1 : -1]
    targets = response_ids
    mask = response_loss_mask.clone()
    if mask.shape[1]:
        mask[:, 0] = False
    if response_logits.shape[1] != targets.shape[1]:
        raise ValueError("logits do not contain a full response window")
    selected = response_logits[mask]
    labels = targets[mask]
    if labels.numel() == 0:
        return {"token_accuracy": 0.0, "perplexity": float("inf"), "tokens": 0.0}
    loss = torch.nn.functional.cross_entropy(selected.float(), labels, reduction="mean")
    accuracy = (selected.argmax(dim=-1) == labels).float().mean()
    return {"token_accuracy": float(accuracy), "perplexity": float(torch.exp(loss)), "tokens": float(labels.numel())}


def validate_modality_contracts(contracts: list[object]) -> None:
    """Require explicit, separately-versioned contracts for non-image modalities."""
    for contract in contracts:
        if not hasattr(contract, "validate"): raise ValueError("invalid modality contract")
        contract.validate()
    modalities=[getattr(c,"modality",None) for c in contracts]
    if len(modalities) != len(set(modalities)): raise ValueError("duplicate modality contracts")
