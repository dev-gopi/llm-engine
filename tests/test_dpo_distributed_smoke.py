import os
import socket

import pytest
import torch
import torch.multiprocessing as mp

from model.gpt import MiniGPT
from post_training.dpo import DPOTrainer
from training.distributed import DistributedTrainer


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _worker(rank: int, world_size: int, port: int, queue) -> None:
    os.environ.update({
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
        "WORLD_SIZE": str(world_size),
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
    })
    torch.manual_seed(7)
    context = DistributedTrainer.initialize("gloo")
    policy = MiniGPT(vocab_size=32, dim=8, layers=1, heads=2, max_pos=8)
    reference = MiniGPT(vocab_size=32, dim=8, layers=1, heads=2, max_pos=8)
    reference.load_state_dict(policy.state_dict())
    wrapped = DistributedTrainer.wrap(policy, context, strategy="ddp")
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
    trainer = DPOTrainer(
        wrapped, reference, optimizer, method="dpo", distributed_context=context,
        gradient_clip_norm=1.0,
    )
    chosen = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    rejected = torch.tensor([[1, 2, 5, 6]], dtype=torch.long)
    attention = torch.ones_like(chosen, dtype=torch.bool)
    response_mask = torch.tensor([[False, True, True]], dtype=torch.bool)
    metrics = trainer.train_step({
        "chosen_ids": chosen,
        "rejected_ids": rejected,
        "chosen_attention_mask": attention,
        "rejected_attention_mask": attention,
        "chosen_mask": response_mask,
        "rejected_mask": response_mask,
    })
    checksum = sum(float(p.detach().sum()) for p in policy.parameters())
    gathered = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, checksum)
    if rank == 0:
        queue.put((metrics["loss"], gathered))
    DistributedTrainer.shutdown()


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed unavailable")
def test_dpo_ddp_two_process_cpu_smoke() -> None:
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    mp.spawn(_worker, args=(2, _free_port(), queue), nprocs=2, join=True)
    loss, checksums = queue.get()
    assert loss > 0
    assert checksums[0] == pytest.approx(checksums[1], abs=1e-6)
