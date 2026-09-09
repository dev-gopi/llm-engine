import torch
from torch import nn

from optim.adamw import build_adamw
from optim.ema import EMA
from optim.scheduler import Scheduler


def test_adamw_groups_scheduler_and_ema() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    optimizer = build_adamw(model, learning_rate=1e-3, weight_decay=0.1)
    assert {group["weight_decay"] for group in optimizer.param_groups} == {0.0, 0.1}
    scheduler = Scheduler(optimizer, warmup_steps=1, total_steps=4, min_lr_ratio=0.1)
    ema = EMA(model, decay=0.5)
    before = {name: value.clone() for name, value in ema.shadow.items()}
    model(torch.ones(2, 4)).sum().backward()
    optimizer.step()
    scheduler.step()
    ema.update(model)
    assert ema.num_updates == 1
    assert any(not torch.equal(before[name], value) for name, value in ema.shadow.items())


def test_ema_average_context_restores_parameters() -> None:
    model = nn.Linear(2, 2)
    ema = EMA(model, decay=0.9)
    original = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(1)
    changed = model.weight.detach().clone()
    with ema.average_parameters(model):
        torch.testing.assert_close(model.weight, original)
    torch.testing.assert_close(model.weight, changed)


def test_ema_average_context_supports_cpu_backup() -> None:
    model = nn.Linear(2, 2)
    ema = EMA(model, decay=0.9)
    original = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(1)
    changed = model.weight.detach().clone()
    with ema.average_parameters(model, backup_device="cpu"):
        torch.testing.assert_close(model.weight, original)
    torch.testing.assert_close(model.weight, changed)


def test_auto_fused_optimizer_uses_cpu_fallback():
    from optim.adamw import adamw_from_config
    model = nn.Linear(2, 2)
    optimizer = adamw_from_config(model, {"fused_optimizer": "auto"})
    assert optimizer.defaults["fused"] is False
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()


def test_invalid_fused_optimizer_is_rejected():
    import pytest
    with pytest.raises(ValueError, match="fused_optimizer"):
        build_adamw(nn.Linear(2, 2), fused="yes")
