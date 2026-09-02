import pytest
import torch

from model.gpt import MiniGPT
from multimodal.model import VisionLanguageModel
from vision.encoder import VisionEncoder
from vision.patch_embedding import PatchEmbedding


def small_vision() -> VisionEncoder:
    return VisionEncoder(
        image_size=32,
        patch_size=8,
        hidden_size=24,
        layers=2,
        heads=3,
        ffn_hidden_size=48,
        dropout=0.0,
    )


def test_patch_embedding_returns_spatial_token_sequence() -> None:
    module = PatchEmbedding(32, 8, 3, 24)
    output = module(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 16, 24)


def test_vision_encoder_returns_class_and_patch_tokens() -> None:
    output = small_vision()(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 17, 24)
    assert torch.isfinite(output).all()


def test_vision_encoder_rejects_wrong_image_size() -> None:
    with pytest.raises(ValueError, match="trailing shape"):
        small_vision()(torch.randn(1, 3, 16, 16))


def test_multimodal_wrapper_keeps_base_models_frozen_and_projector_trainable() -> None:
    language = MiniGPT(
        vocab_size=32,
        dim=16,
        layers=1,
        heads=2,
        max_pos=32,
        position_type="rotary",
    )
    model = VisionLanguageModel(small_vision(), language, visual_tokens=4)
    assert not any(parameter.requires_grad for parameter in model.vision_encoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.language_model.parameters())
    assert all(parameter.requires_grad for parameter in model.projector.parameters())
    model.train()
    assert model.projector.training
    assert not model.vision_encoder.training
    assert not model.language_model.training

    logits, loss_mask = model(
        torch.randn(2, 3, 32, 32),
        torch.tensor([[1, 2, 3], [4, 5, 6]]),
        torch.tensor([[7, 8], [9, 10]]),
    )
    assert logits.shape == (2, 9, 32)
    assert loss_mask.shape == (2, 9)
    assert loss_mask.sum().item() == 4
    assert torch.isfinite(logits).all()


def test_multimodal_wrapper_does_not_change_language_state_keys() -> None:
    language = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=16)
    keys_before = tuple(language.state_dict())
    VisionLanguageModel(small_vision(), language, visual_tokens=2)
    assert tuple(language.state_dict()) == keys_before
