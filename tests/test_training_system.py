import torch
from torch.utils.data import DataLoader

from datasets.collator import Collator
from model.gpt import MiniGPT
from model.loss import CausalLanguageModelLoss
from optim.adamw import build_adamw
from optim.ema import EMA
from optim.scheduler import Scheduler
from training.checkpoint import load_checkpoint, save_checkpoint
from training.distributed import DistributedTrainer
from training.evaluator import Evaluator
from training.trainer import Trainer


def make_loader():
    return DataLoader(
        [torch.tensor([1, 2, 3, 4]), torch.tensor([2, 3, 4])],
        batch_size=2,
        collate_fn=Collator(0),
    )


def test_fit_evaluate_checkpoint_and_resume(tmp_path) -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    optimizer = build_adamw(model, learning_rate=1e-3)
    scheduler = Scheduler(optimizer, warmup_steps=0, total_steps=2)
    ema = EMA(model, decay=0.9)
    loss_fn = CausalLanguageModelLoss(shift_labels=True)
    trainer = Trainer(model, optimizer, loss_fn, scheduler=scheduler, ema=ema)
    evaluator = Evaluator(model, loss_fn=loss_fn)
    history = trainer.fit(make_loader(), epochs=1, evaluator=evaluator, validation_dataloader=make_loader(), log_every=0)
    assert history[-1]["tokens"] == 5
    assert history[-1]["perplexity"] > 0

    path = save_checkpoint(tmp_path / "model.pt", model, optimizer=optimizer, scheduler=scheduler, ema=ema, step=trainer.global_step)
    restored = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    restored_optimizer = build_adamw(restored)
    restored_scheduler = Scheduler(restored_optimizer, warmup_steps=0, total_steps=2)
    restored_ema = EMA(restored)
    state = load_checkpoint(path, restored, optimizer=restored_optimizer, scheduler=restored_scheduler, ema=restored_ema)
    assert state["step"] == 1
    assert restored_ema.num_updates == 1
    assert restored(torch.tensor([[1, 2]])).shape == (1, 2, 16)


def test_single_process_distributed_helpers() -> None:
    context = DistributedTrainer.initialize()
    assert context.world_size == 1
    value = torch.tensor(2.0)
    assert DistributedTrainer.mean(value, context).item() == 2.0


def test_checkpoint_can_apply_ema_weights(tmp_path) -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    ema = EMA(model, decay=0.9)
    expected = ema.shadow["tok.embedding.weight"].clone()
    with torch.no_grad():
        model.tok.weight.add_(2)
    path = save_checkpoint(tmp_path / "ema.pt", model, ema=ema)
    restored = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    state = load_checkpoint(path, restored, use_ema=True)
    assert state["ema_applied"]
    torch.testing.assert_close(restored.tok.weight, expected)
