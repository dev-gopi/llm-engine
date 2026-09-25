import pytest
import torch

from model.gpt import MiniGPT
from training.model_growth import grow_model, identity_output_error


def make_model(vocab_size: int, layers: int) -> MiniGPT:
    return MiniGPT(
        vocab_size=vocab_size,
        dim=16,
        layers=layers,
        heads=4,
        kv_heads=2,
        max_pos=16,
        ffn_hidden_dim=32,
        ffn_activation="swiglu",
        norm_type="rms_norm",
        norm_bias=False,
        attention_bias=False,
        ffn_bias=False,
        position_type="rotary",
        tie_word_embeddings=True,
    )


def test_grow_model_preserves_existing_logits_and_initializes_identity_layers() -> None:
    torch.manual_seed(7)
    source = make_model(32, 2).eval()
    target = make_model(36, 4).eval()
    token_ids = torch.tensor([[1, 2, 3]])
    expected = source(token_ids)

    report = grow_model(source, target)
    actual = target(token_ids)

    torch.testing.assert_close(actual[..., :32], expected)
    torch.testing.assert_close(target.tok.weight[:32], source.tok.weight)
    expected_new_rows = source.tok.weight.mean(dim=0).expand(4, -1)
    torch.testing.assert_close(target.tok.weight[32:], expected_new_rows)
    for block in target.blocks[2:]:
        assert torch.count_nonzero(block.attn.out_proj.weight) == 0
        assert torch.count_nonzero(block.ffn.out_proj.weight) == 0
    assert report.new_parameters == target.num_parameters() - source.num_parameters()


def test_grow_model_rejects_non_growth_and_architecture_mismatch() -> None:
    with pytest.raises(ValueError, match="more transformer layers"):
        grow_model(make_model(32, 2), make_model(36, 2))
    incompatible = MiniGPT(
        vocab_size=36, dim=24, layers=3, heads=4, kv_heads=2,
        max_pos=16, ffn_hidden_dim=32,
    )
    with pytest.raises(ValueError, match="architecture mismatch"):
        grow_model(make_model(32, 2), incompatible)


def test_depth_doubling_from_sixteen_to_thirty_two_layers_is_an_exact_identity() -> None:
    torch.manual_seed(9)
    source = make_model(32, 16)
    target = make_model(32, 32)
    grow_model(source, target)

    assert identity_output_error(source, target, torch.tensor([[1, 2, 3, 4]])) == 0.0


def test_grow_model_supports_mtp_and_moe() -> None:
    torch.manual_seed(11)
    source_moe = MiniGPT(
        vocab_size=32, dim=16, layers=2, heads=4, kv_heads=2, max_pos=16,
        ffn_type="moe", num_experts=4, experts_per_token=2, mtp_num_predictions=2,
    ).eval()
    target_moe = MiniGPT(
        vocab_size=36, dim=16, layers=4, heads=4, kv_heads=2, max_pos=16,
        ffn_type="moe", num_experts=4, experts_per_token=2, mtp_num_predictions=2,
    ).eval()
    report = grow_model(source_moe, target_moe)
    assert report.source_vocab_size == 32
    assert report.target_vocab_size == 36
    for block in target_moe.blocks[2:]:
        assert torch.count_nonzero(block.attn.out_proj.weight) == 0
        for expert in block.ffn.experts:
            assert torch.count_nonzero(expert.out_proj.weight) == 0

