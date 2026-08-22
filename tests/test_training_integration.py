import torch

from datasets.collator import Collator
from model.gpt import MiniGPT
from optim.adamw import build_adamw
from optim.ema import EMA
from optim.scheduler import Scheduler
from training.trainer import Trainer


def test_dataset_batch_trains_model_optimizer_scheduler_and_ema() -> None:
    model = MiniGPT(vocab_size=32, dim=8, layers=1, heads=2, max_pos=8)
    batch = Collator(pad_token_id=0)([
        torch.tensor([1, 2, 3, 4]),
        torch.tensor([5, 6, 7]),
    ])
    optimizer = build_adamw(model, learning_rate=1e-3)
    scheduler = Scheduler(optimizer, warmup_steps=0, total_steps=3)
    ema = EMA(model, decay=0.9)
    trainer = Trainer(model, optimizer, scheduler=scheduler, ema=ema)
    loss = trainer.train_step(batch)
    assert loss > 0
    assert ema.num_updates == 1
    assert scheduler.last_epoch == 1


def test_gradient_accumulation_steps_optimizer_once() -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    optimizer = build_adamw(model, learning_rate=1e-3)
    trainer = Trainer(model, optimizer, gradient_accumulation_steps=2)
    batch = Collator(0)([torch.tensor([1, 2, 3])])
    before = model.tok.weight.detach().clone()
    trainer.train_step(batch)
    assert trainer.global_step == 0
    torch.testing.assert_close(model.tok.weight, before)
    trainer.train_step(batch)
    assert trainer.global_step == 1
    assert not torch.equal(model.tok.weight, before)
