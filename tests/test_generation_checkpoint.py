import pytest
import torch

from optim.ema import EMA
from training.generation_checkpoint import retention_passes, save_best_generation
from training.checkpoint import load_checkpoint


def test_preserves_better_answers_across_restart_and_ties(tmp_path):
    model = torch.nn.Linear(2, 2)
    path = tmp_path / 'best.pt'
    original = model.weight.detach().clone()
    def save(score, step, signature='fixed-suite'):
        return save_best_generation(path, model, accuracy=score,
                                    evaluation_signature=signature, step=step, metadata={})
    assert save(3 / 18, 2000)
    with torch.no_grad():
        model.weight.add_(1)
    assert not save(2 / 18, 4000)
    assert not save(3 / 18, 6000)
    payload = torch.load(path, weights_only=True)
    assert payload['step'] == 2000
    torch.testing.assert_close(payload['model']['weight'], original)
    with pytest.raises(ValueError, match='evaluation changed'):
        save(1, 8000, 'different-suite')
    assert save(4 / 18, 10000)
    assert torch.load(path, weights_only=True)['step'] == 10000


def test_saves_exact_evaluated_ema_weights(tmp_path):
    model = torch.nn.Linear(2, 2)
    ema = EMA(model, decay=0.9)
    expected = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(2)
    path = tmp_path / 'ema.pt'
    with ema.average_parameters(model):
        save_best_generation(path, model, accuracy=0.5,
                             evaluation_signature='suite', step=2, metadata={})
    torch.testing.assert_close(model.weight, expected + 2)
    loaded = torch.nn.Linear(2, 2)
    load_checkpoint(path, loaded, use_ema=True, restore_rng=False)
    torch.testing.assert_close(loaded.weight, expected)


def test_retention_rejects_higher_average_that_loses_a_preserved_case(tmp_path):
    model = torch.nn.Linear(2, 2)
    path = tmp_path / "retained.pt"
    def save(accuracy, scores):
        return save_best_generation(path, model, accuracy=accuracy,
                                    evaluation_signature="fixed", step=1, metadata={},
                                    case_scores=scores, preserve_passed=True)
    assert save(1 / 3, {"en": 1, "bn": 0, "hi": 0})
    assert not save(2 / 3, {"en": 0, "bn": 1, "hi": 1})
    assert save(2 / 3, {"en": 1, "bn": 1, "hi": 0})
    assert not save(2 / 3, {"en": 1, "bn": 0, "hi": 1})


def test_retention_gate_can_protect_the_separate_loss_best_checkpoint(tmp_path):
    model = torch.nn.Linear(2, 2)
    path = tmp_path / "retained.pt"
    baseline = {"en": 1, "bn": 0}
    assert save_best_generation(
        path, model, accuracy=0.5, evaluation_signature="fixed", step=0,
        metadata={}, case_scores=baseline, preserve_passed=True,
    )

    assert retention_passes(
        path, evaluation_signature="fixed", case_scores={"en": 1, "bn": 1}
    )
    assert not retention_passes(
        path, evaluation_signature="fixed", case_scores={"en": 0, "bn": 1}
    )


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -1, 2])
def test_rejects_invalid_score(tmp_path, score):
    with pytest.raises(ValueError, match='accuracy'):
        save_best_generation(tmp_path / 'best.pt', torch.nn.Linear(1, 1),
                             accuracy=score, evaluation_signature='suite', step=1, metadata={})
