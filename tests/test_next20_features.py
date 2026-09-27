from pathlib import Path

import torch
from torch import nn

from inference.beam_search import beam_search
from inference.parallel_runtime import (
    ParallelRuntimeConfig,
    sequence_shard,
    split_pipeline_blocks,
)
from model.expert_paging import ExpertPager
from model.seq2seq import EncoderDecoderTransformer
from post_training.distributed_online_grpo import (
    async_rollouts,
    recover_after_rank_failure,
    shard_prompts,
)
from post_training.ppo import (
    AdaptiveKLController,
    PPOTrainer,
    ValueHead,
    ValueModel,
    compute_gae,
    load_value_checkpoint,
    ppo_clipped_loss,
    save_value_checkpoint,
)
from post_training.reward_monitor import RewardDriftMonitor
from post_training.reward_objectives import (
    listwise_reward_loss,
    pointwise_regression_loss,
)
from post_training.verifiers import CompositeVerifier, ExactMatchVerifier
from serving.pipeline_parallel import PipelineServingRuntime
from training.context_parallel import shard_batch_sequence
from training.deepspeed import ZeROConfig, build_config
from training.experiment_tracking import NullTracker
from training.expert_parallel import expert_indices
from training.pipeline_parallel import PipelineSchedule, run_pipeline


class TinyLM(nn.Module):
    def __init__(self, vocab=16, hidden=8):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, ids, attention_mask=None, return_hidden_states=False):
        h = self.emb(ids)
        logits = self.head(h)
        return (logits, h) if return_hidden_states else logits


def test_gae_and_adaptive_kl():
    rewards = torch.tensor([[1.0, 0.0, 2.0]])
    values = torch.zeros_like(rewards)
    adv, ret = compute_gae(rewards, values)
    assert adv.shape == rewards.shape and torch.allclose(ret, adv)
    c = AdaptiveKLController(coefficient=0.1, target=0.1, horizon=10)
    assert c.update(0.2) > 0.1


def test_ppo_loss_and_value_checkpoint(tmp_path):
    new = torch.tensor([-0.2, -0.4], requires_grad=True)
    old = torch.tensor([-0.3, -0.3])
    adv = torch.tensor([1.0, -1.0])
    ret = torch.tensor([0.5, -0.2])
    val = torch.tensor([0.1, 0.0], requires_grad=True)
    loss, m = ppo_clipped_loss(new, old, adv, ret, val)
    loss.backward()
    assert torch.isfinite(loss)
    vm = ValueHead(4)
    opt = torch.optim.AdamW(vm.parameters(), lr=1e-3)
    p = save_value_checkpoint(tmp_path / "v.pt", vm, opt, step=3)
    assert p.exists()
    load_value_checkpoint(p, vm, opt)


def test_ppo_trainer_single_update():
    actor = TinyLM()
    ref = TinyLM()
    reward = TinyLM()
    value = ValueModel(TinyLM(), 8)
    opt = torch.optim.AdamW(actor.parameters(), lr=1e-3)
    vopt = torch.optim.AdamW(value.parameters(), lr=1e-3)
    trainer = PPOTrainer(actor, value, ref, reward, opt, value_optimizer=vopt)
    ids = torch.randint(0, 16, (3, 5))
    mask = torch.ones(3, 4, dtype=torch.bool)
    out = trainer.update(
        {
            "token_ids": ids,
            "old_log_probs": torch.zeros(3),
            "advantages": torch.ones(3),
            "returns": torch.zeros(3),
            "mask": mask,
        }
    )
    assert out["step"] == 1


def test_reward_objectives_monitor():
    loss, _ = pointwise_regression_loss(
        torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0])
    )
    assert loss.item() >= 0
    order = torch.tensor([[0, 1, 2], [2, 1, 0]])
    scores = torch.randn(2, 3)
    loss, _ = listwise_reward_loss(scores, order)
    assert torch.isfinite(loss)
    mon = RewardDriftMonitor()
    assert not mon.observe("v1", [0.0, 1.0])["alert"]
    mon.observe("v2", [10.0, 11.0])
    mon.promote("v2")
    assert mon.promoted_version == "v2"


def test_verifiers_and_async_rollouts():
    assert ExactMatchVerifier().verify("p", "yes", {"expected_answer": "yes"}) == 1
    assert (
        CompositeVerifier([ExactMatchVerifier()]).verify(
            "p", "yes", {"expected_answer": "yes"}
        )
        == 1
    )
    success, fail = async_rollouts([1, 2], lambda x: {"prompt_index": x, "x": x * x})
    assert len(success) == 2 and not fail
    assert shard_prompts(range(6), 1, 2) == [1, 3, 5]
    assert recover_after_rank_failure(1, 2, ["x"])["replacement_rank"] == 0


def test_parallel_contracts():
    ParallelRuntimeConfig(context_parallel=2).validate(world_size=2, sequence_length=8)
    x = torch.arange(16).reshape(1, 8, 2)
    assert sequence_shard(x, 1, 2).shape[1] == 4
    assert (
        len(split_pipeline_blocks(nn.ModuleList([nn.Identity() for _ in range(4)]), 2))
        == 2
    )
    assert (
        shard_batch_sequence({"input_ids": x}, rank=1, world_size=2)["input_ids"].shape[
            1
        ]
        == 4
    )
    assert expert_indices(8, 1, 2) == (1, 3, 5, 7)


def test_pipeline_deepspeed_and_tracking():
    sched = PipelineSchedule(2, 3)
    assert sched.one_forward_one_backward()
    assert run_pipeline([lambda x: x + 1, lambda x: x * 2], 2, [1, 2]) == [4, 6]
    cfg = build_config(ZeROConfig(stage=2), micro_batch_size=2)
    assert cfg["zero_optimization"]["stage"] == 2
    NullTracker().log({"x": 1}, step=1)
    pager = ExpertPager(Path("tmp-experts"), max_resident=1)
    pager.store(0, {"w": torch.ones(2)})
    assert 0 in pager.available_indices()
    assert pager.load(0)["w"].sum() == 2
    runtime = PipelineServingRuntime(
        [lambda x: x + 1, lambda x: x * 2], max_in_flight=1
    )
    assert runtime.generate(2) == 6
    runtime.close()


def test_seq2seq_and_beam_search():
    model = EncoderDecoderTransformer(
        vocab_size=20, d_model=16, nhead=4, layers=1, ff_dim=32, max_position=16
    )
    src = torch.randint(0, 20, (2, 4))
    tgt = torch.randint(0, 20, (2, 3))
    out = model(src, tgt)
    assert out.shape == (2, 3, 20)

    def logits(ids):
        x = torch.zeros(1, ids.shape[1], 5)
        x[:, :, -1] = 1
        return x

    result = beam_search(logits, torch.tensor([0]), num_beams=2, max_new_tokens=2)
    assert len(result) == 3


def test_supported_profiles_no_longer_planning_only():
    for name in [
        "configs/pretraining.packed.gpu.yaml",
        "configs/pretraining.packed.cpu.yaml",
        "configs/finetuning.packed.gpu.yaml",
        "configs/finetuning.packed.cpu.yaml",
        "configs/vision/training.production.yaml",
        "configs/vision/training.local.yaml",
        "configs/diffusion/training.production.yaml",
        "configs/diffusion/training.local.yaml",
        "configs/corpus.web.yaml",
    ]:
        text = Path(name).read_text()
        assert "planning_only: true" not in text
