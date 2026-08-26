import asyncio
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from datasets.filters import CorpusFilter
from datasets.token_shards import TokenShardDataset
from evaluation.benchmarks import BenchmarkCase, score_answer, summarize_scores
from inference.paged_kv_cache import PagedKVCache, PrefixCache
from model.gpt import MiniGPT
from post_training.dpo import DPOLoss, sequence_log_probabilities
from serving.batching import DynamicBatcher
from training.distributed import DistributedContext, DistributedTrainer
from training.distributed_checkpoint import load_distributed_checkpoint, save_distributed_checkpoint


def test_corpus_filter_deduplicates_and_redacts_pii() -> None:
    corpus_filter = CorpusFilter(min_chars=5)
    text = corpus_filter.apply("Contact person@example.com or +1 212 555 0100 today")
    assert text is not None and "<email>" in text and "<phone>" in text
    assert corpus_filter.apply("Contact person@example.com or +1 212 555 0100 today") is None
    assert corpus_filter.stats.duplicate == 1


def test_binary_token_shard_dataset(tmp_path) -> None:
    values = np.arange(24, dtype=np.uint32).reshape(3, 8)
    values.tofile(tmp_path / "tokens-00000.bin")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "format": "gopi-token-shards-v1",
        "dtype": "uint32",
        "sequence_length": 8,
        "shards": [{"file": "tokens-00000.bin", "sequences": 3}],
    }))
    dataset = TokenShardDataset(tmp_path / "manifest.json")
    assert len(dataset) == 3 and dataset.lengths == [8, 8, 8]
    torch.testing.assert_close(dataset[1]["input_ids"], torch.arange(8, 16))
    (tmp_path / "tokens-00000.bin").write_bytes(b"bad")
    with pytest.raises(ValueError, match="size mismatch"):
        TokenShardDataset(tmp_path / "manifest.json")


def test_token_shard_cli_does_not_shadow_standard_library_tokenize() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/build_token_shards.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--sequence-length" in completed.stdout


def test_paged_cache_round_trip_and_prefix_lru() -> None:
    cache = PagedKVCache(
        num_pages=3, page_size=2, layers=2, kv_heads=1, head_dim=4,
        device="cpu", dtype=torch.float32,
    )
    cache.reserve("a", 3)
    keys = torch.randn(2, 1, 3, 4)
    values = torch.randn(2, 1, 3, 4)
    cache.append("a", keys, values)
    actual_keys, actual_values = cache.materialize("a")
    torch.testing.assert_close(actual_keys, keys)
    torch.testing.assert_close(actual_values, values)
    cache.release("a")
    with pytest.raises(ValueError, match="positive integer"):
        cache.reserve("bad", 0)
    with pytest.raises(KeyError, match="unknown request"):
        cache.materialize("missing")
    prefixes = PrefixCache(capacity=1)
    prefixes.put((1,), "first")
    prefixes.put((2,), "second")
    assert prefixes.get((1,)) is None and prefixes.get((2,)) == "second"


def test_dpo_loss_and_sequence_scores() -> None:
    logits = torch.zeros(2, 4, 8)
    tokens = torch.tensor([[1, 2, 3, 4], [1, 3, 4, 5]])
    mask = torch.ones(2, 3, dtype=torch.bool)
    scores = sequence_log_probabilities(logits, tokens, mask)
    assert scores.shape == (2,)
    loss, metrics = DPOLoss(beta=0.1)(
        torch.tensor([2.0]), torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([1.0])
    )
    assert loss.item() < 0.7 and metrics["reward_accuracy"].item() == 1


def test_dynamic_batcher_groups_concurrent_requests() -> None:
    class Backend:
        batches = []

        async def batch_generate(self, requests):
            self.batches.append(list(requests))
            return [value * 2 for value in requests]

    async def scenario():
        backend = Backend()
        batcher = DynamicBatcher(backend, max_batch_size=4, wait_milliseconds=2)
        await batcher.startup()
        results = await asyncio.gather(*(batcher.generate(value) for value in range(3)))
        await batcher.shutdown()
        return backend, results

    backend, results = asyncio.run(scenario())
    assert results == [0, 2, 4]
    assert backend.batches == [[0, 1, 2]]


def test_benchmark_scoring_by_category() -> None:
    case = BenchmarkCase("math", "2+2", ("4",), ("5",))
    results = [(case, score_answer("The answer is 4.", case))]
    assert summarize_scores(results) == {"cases": 1, "accuracy": 1.0, "accuracy_math": 1.0}


def test_default_single_process_distributed_behavior_is_unchanged() -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2)
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    assert DistributedTrainer.wrap(model, context, strategy="ddp") is model
    with pytest.raises(ValueError, match="distributed_strategy"):
        DistributedTrainer.wrap(model, context, strategy="invalid")


def test_distributed_checkpoint_round_trip_without_process_group(tmp_path) -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    original = model.tok.weight.detach().clone()
    with pytest.warns(UserWarning, match="assuming the intent is to save in a single process"):
        path = save_distributed_checkpoint(
            tmp_path / "dcp", model, optimizer, metadata={"step": 7}
        )
    # Periodic checkpoints reuse the configured destination safely.
    with pytest.warns(UserWarning) as warnings:
        path = save_distributed_checkpoint(path, model, optimizer, metadata={"step": 8})
    messages = [str(warning.message) for warning in warnings]
    assert any("assuming the intent is to save in a single process" in message for message in messages)
    assert any("overwriting since self.overwrite=True" in message for message in messages)
    with torch.no_grad():
        model.tok.weight.add_(1)
    with pytest.warns(UserWarning, match="assuming the intent is to load in a single process"):
        metadata = load_distributed_checkpoint(path, model, optimizer)
    torch.testing.assert_close(model.tok.weight, original)
    assert metadata["step"] == 8
    assert metadata["trainer"] == {} and metadata["sampler"] == {}
