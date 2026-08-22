import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.loss import CausalLanguageModelLoss, LanguageModelLossOutput
from training.metrics import perplexity
from training.trainer import Trainer


def test_shifted_loss_matches_pytorch_reference():
    torch.manual_seed(40)
    logits = torch.randn(2, 6, 13)
    labels = torch.randint(0, 13, (2, 6))
    expected = F.cross_entropy(logits[:, :-1].reshape(-1, 13), labels[:, 1:].reshape(-1))
    actual = CausalLanguageModelLoss()(logits, labels)
    torch.testing.assert_close(actual, expected)


def test_unshifted_targets_match_reference():
    logits = torch.randn(2, 5, 11)
    targets = torch.randint(0, 11, (2, 5))
    expected = F.cross_entropy(logits.reshape(-1, 11), targets.reshape(-1))
    actual = CausalLanguageModelLoss(shift_labels=False)(logits, targets)
    torch.testing.assert_close(actual, expected)


def test_ignore_index_and_loss_mask_exclude_tokens():
    logits = torch.randn(2, 5, 7)
    labels = torch.randint(0, 7, (2, 5))
    labels[0, 2] = -100
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
    details = CausalLanguageModelLoss(shift_labels=False)(
        logits, labels, loss_mask=mask, return_details=True
    )
    selected = mask & labels.ne(-100)
    expected = F.cross_entropy(logits[selected], labels[selected])
    assert isinstance(details, LanguageModelLossOutput)
    assert details.token_count == selected.sum().item()
    torch.testing.assert_close(details.loss, expected)
    torch.testing.assert_close(details.cross_entropy, expected)


def test_label_smoothing_matches_reference():
    logits = torch.randn(2, 4, 9)
    labels = torch.randint(0, 9, (2, 4))
    expected = F.cross_entropy(
        logits.reshape(-1, 9), labels.reshape(-1), label_smoothing=0.1
    )
    actual = CausalLanguageModelLoss(
        shift_labels=False, label_smoothing=0.1
    )(logits, labels)
    torch.testing.assert_close(actual, expected)


def test_z_loss_matches_explicit_formula():
    coefficient = 1e-3
    logits = torch.randn(2, 4, 9)
    labels = torch.randint(0, 9, (2, 4))
    details = CausalLanguageModelLoss(
        shift_labels=False, z_loss_coefficient=coefficient
    )(logits, labels, return_details=True)
    expected_ce = F.cross_entropy(logits.reshape(-1, 9), labels.reshape(-1))
    expected_z = torch.logsumexp(logits.reshape(-1, 9), dim=-1).square().mean()
    torch.testing.assert_close(details.cross_entropy, expected_ce)
    torch.testing.assert_close(details.z_loss, expected_z)
    torch.testing.assert_close(details.loss, expected_ce + coefficient * expected_z)


def test_sum_reduction_returns_token_sum():
    logits = torch.randn(2, 4, 9)
    labels = torch.randint(0, 9, (2, 4))
    expected = F.cross_entropy(logits.reshape(-1, 9), labels.reshape(-1), reduction="sum")
    actual = CausalLanguageModelLoss(shift_labels=False, reduction="sum")(logits, labels)
    torch.testing.assert_close(actual, expected)


def test_all_ignored_batch_returns_differentiable_zero():
    logits = torch.randn(2, 4, 9, requires_grad=True)
    labels = torch.full((2, 4), -100)
    details = CausalLanguageModelLoss(shift_labels=False)(
        logits, labels, return_details=True
    )
    assert details.token_count == 0
    assert details.loss.item() == 0
    assert torch.isfinite(details.loss)
    details.loss.backward()
    assert logits.grad is not None
    assert logits.grad.count_nonzero().item() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_logits_use_float32_loss(dtype):
    logits = torch.randn(2, 4, 9, dtype=dtype, requires_grad=True)
    labels = torch.randint(0, 9, (2, 4))
    loss = CausalLanguageModelLoss(shift_labels=False)(logits, labels)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_from_training_config():
    loss_fn = CausalLanguageModelLoss.from_config(
        {
            "ignore_index": -1,
            "label_smoothing": 0.2,
            "z_loss_coefficient": 1e-4,
            "shift_labels": False,
            "loss_reduction": "sum",
        }
    )
    assert loss_fn.ignore_index == -1
    assert loss_fn.label_smoothing == 0.2
    assert loss_fn.z_loss_coefficient == 1e-4
    assert not loss_fn.shift_labels
    assert loss_fn.reduction == "sum"


def test_perplexity_is_exponential_cross_entropy():
    assert perplexity(math.log(10)) == pytest.approx(10)
    torch.testing.assert_close(perplexity(torch.tensor(math.log(5))), torch.tensor(5.0))


class TinyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.output = nn.Linear(8, 16)

    def forward(self, token_ids):
        return self.output(self.embedding(token_ids))


def test_trainer_uses_production_loss_and_updates_parameters():
    model = TinyLanguageModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, optimizer)
    inputs = torch.randint(0, 16, (2, 5))
    targets = torch.randint(0, 16, (2, 5))
    before = model.output.weight.detach().clone()
    loss = trainer.train_step(inputs, targets)
    assert math.isfinite(loss)
    assert not torch.equal(model.output.weight, before)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"label_smoothing": -0.1},
        {"label_smoothing": 1.0},
        {"z_loss_coefficient": -1},
        {"reduction": "none"},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        CausalLanguageModelLoss(**kwargs)


def test_invalid_inputs_and_labels():
    loss_fn = CausalLanguageModelLoss(shift_labels=False)
    with pytest.raises(ValueError, match="logits"):
        loss_fn(torch.randn(2, 9), torch.ones(2, dtype=torch.long))
    with pytest.raises(ValueError, match="match"):
        loss_fn(torch.randn(2, 4, 9), torch.ones(2, 3, dtype=torch.long))
    with pytest.raises(TypeError, match="int32 or torch.int64"):
        loss_fn(torch.randn(2, 4, 9), torch.ones(2, 4))
    invalid_labels = torch.zeros(2, 4, dtype=torch.long)
    invalid_labels[0, 0] = 9
    with pytest.raises(ValueError, match="labels"):
        loss_fn(torch.randn(2, 4, 9), invalid_labels)


def test_shift_requires_two_tokens():
    with pytest.raises(ValueError, match="at least two"):
        CausalLanguageModelLoss()(torch.randn(2, 1, 9), torch.ones(2, 1, dtype=torch.long))
