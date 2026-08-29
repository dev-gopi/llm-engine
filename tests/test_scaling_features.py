import asyncio
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from datasets.filters import CorpusFilter
from datasets.token_shards import TokenShardDataset
from evaluation.benchmarks import BenchmarkCase, normalize_answer, score_answer, summarize_scores
from inference.paged_kv_cache import PagedKVCache, PrefixCache
from model.gpt import MiniGPT
from post_training.dpo import DPOLoss, sequence_log_probabilities
from scripts.evaluate_domains import aggregate_domain_metrics
from serving.batching import DynamicBatcher
from training.distributed import DistributedContext, DistributedTrainer
from training.distributed_checkpoint import load_distributed_checkpoint, save_distributed_checkpoint


def test_corpus_filter_deduplicates_and_redacts_pii() -> None:
    corpus_filter = CorpusFilter(min_chars=5)
    text = corpus_filter.apply("Contact person@example.com or +1 212 555 0100 today")
    assert text is not None and "<email>" in text and "<phone>" in text
    assert corpus_filter.apply("Contact person@example.com or +1 212 555 0100 today") is None
    assert corpus_filter.stats.duplicate == 1


def test_corpus_filter_rejects_near_duplicates_and_benchmark_contamination() -> None:
    benchmark = (
        "The patient benchmark answer explains how rainfall forms over mountains "
        "during a cold winter afternoon."
    )
    corpus_filter = CorpusFilter(
        min_chars=5, excluded_texts=[benchmark], near_duplicate_distance=3,
    )
    contaminated = benchmark.replace("afternoon", "evening")
    first = (
        "The quick brown fox jumps over the lazy dog beside the quiet river every morning."
    )
    near_duplicate = first.replace("morning", "evening")
    distinct = "A spacecraft uses controlled rocket thrust to adjust its orbit around Mars."

    assert corpus_filter.apply(contaminated) is None
    embedded = (
        "An unrelated introduction appears before this evaluation item. "
        + benchmark
        + " Additional unrelated material follows the leaked answer."
    )
    assert corpus_filter.apply(embedded) is None
    assert corpus_filter.apply(first) is not None
    assert corpus_filter.apply(near_duplicate) is None
    assert corpus_filter.apply(distinct) is not None
    assert corpus_filter.stats.contamination == 2
    assert corpus_filter.stats.near_duplicate == 1


def test_corpus_filter_can_disable_near_duplicate_detection() -> None:
    first = "One two three four five six seven eight nine ten eleven twelve."
    changed = "One two three four five six seven eight nine ten eleven thirteen."
    corpus_filter = CorpusFilter(min_chars=5, near_duplicate_distance=None)
    assert corpus_filter.apply(first) is not None
    assert corpus_filter.apply(changed) is not None


def test_corpus_filter_redacts_validated_structured_pii_and_credentials() -> None:
    text = (
        "Email alice@example.com, phone +1 212 555 0100, IPv4 192.168.1.1, "
        "IPv6 2001:db8::1, SSN 123-45-6789, card 4111 1111 1111 1111, "
        "IBAN GB82 WEST 1234 5698 7654 32, key ghp_abcdefghijklmnopqrstuvwxyz123456, "
        "and address 123 Main Street Apt 4B. Invalid values 999.999.999.999 and "
        "4111 1111 1111 1112 must remain."
    )
    corpus_filter = CorpusFilter(min_chars=5, near_duplicate_distance=None)
    redacted = corpus_filter.apply(text)

    assert redacted is not None
    for placeholder in (
        "<email>", "<phone>", "<ip>", "<government-id>", "<financial-id>",
        "<credential>", "<address>",
    ):
        assert placeholder in redacted
    assert "999.999.999.999" in redacted
    assert "4111 1111 1111 1112" in redacted
    assert corpus_filter.stats.pii_redactions == 9
    assert corpus_filter.stats.ip_redactions == 2
    assert corpus_filter.stats.financial_id_redactions == 2
    assert corpus_filter.stats.credential_redactions == 1


@pytest.mark.parametrize("dtype", [np.uint16, np.uint32])
def test_binary_token_shard_dataset(tmp_path, dtype) -> None:
    values = np.arange(24, dtype=dtype).reshape(3, 8)
    values.tofile(tmp_path / "tokens-00000.bin")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "format": "gopi-token-shards-v1",
        "dtype": np.dtype(dtype).name,
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


def test_paged_prefix_cache_evicts_by_page_pressure() -> None:
    from inference.paged_kv_cache import PagedPrefixCache

    allocator = PagedKVCache(
        num_pages=2, page_size=2, layers=1, kv_heads=1, head_dim=2,
        device="cpu", dtype=torch.float32,
    )
    prefixes = PagedPrefixCache(allocator, capacity=4)
    cache = ((torch.zeros(1, 1, 3, 2), torch.zeros(1, 1, 3, 2)),)
    prefixes.put((1, 2, 3), torch.zeros(1, 3, 8), cache)
    replacement = ((torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2)),)
    prefixes.put((4, 5), torch.zeros(1, 2, 8), replacement)
    assert prefixes.get((1, 2, 3)) is None
    assert prefixes.get((4, 5)) is not None


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


def test_benchmark_normalization_and_scoring_support_unicode_scripts() -> None:
    assert normalize_answer("উত্তর: ঢাকা!") == "উত্তর ঢাকা"
    assert normalize_answer("उत्तर: नई दिल्ली।") == "उत्तर नई दिल्ली"
    assert score_answer(
        "বাংলাদেশের রাজধানী ঢাকা।",
        BenchmarkCase("bengali", "", ("ঢাকা",)),
    ) == 1.0


def test_domain_metrics_are_aggregated_by_token_count() -> None:
    metrics = aggregate_domain_metrics({
        "english": {"loss": 2.0, "cross_entropy": 2.0, "z_loss": 0.0, "tokens": 10, "batches": 1},
        "bengali": {"loss": 4.0, "cross_entropy": 4.0, "z_loss": 0.0, "tokens": 30, "batches": 2},
    })
    assert metrics["loss"] == pytest.approx(3.5)
    assert metrics["cross_entropy"] == pytest.approx(3.5)
    assert metrics["perplexity"] == pytest.approx(np.exp(3.5))
    assert metrics["tokens"] == 40
    assert metrics["batches"] == 3


def test_default_single_process_distributed_behavior_is_unchanged() -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2)
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    assert DistributedTrainer.wrap(model, context, strategy="ddp") is model
    with pytest.raises(ValueError, match="distributed_strategy"):
        DistributedTrainer.wrap(model, context, strategy="invalid")
    multi_process = DistributedContext(0, 0, 2, torch.device("cpu"))
    with pytest.raises(ValueError, match="WORLD_SIZE"):
        DistributedTrainer.wrap(model, multi_process, strategy="none")


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
    assert not any("overwriting since self.overwrite=True" in message for message in messages)
    with torch.no_grad():
        model.tok.weight.add_(1)
    with pytest.warns(UserWarning, match="assuming the intent is to load in a single process"):
        metadata = load_distributed_checkpoint(path, model, optimizer)
    torch.testing.assert_close(model.tok.weight, original)
    assert metadata["step"] == 8
    assert metadata["trainer"] == {} and metadata["sampler"] == {}


def test_distributed_checkpoint_rejects_corruption(tmp_path) -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.warns(UserWarning):
        path = save_distributed_checkpoint(tmp_path / "dcp", model, optimizer)
    rank_state = path / "gopi_rank_00000.pt"
    rank_state.write_bytes(rank_state.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="truncated|checksum"):
        load_distributed_checkpoint(path, model, optimizer)


def test_distributed_checkpoint_restores_scheduler_scaler_and_rng(tmp_path) -> None:
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    optimizer.step()
    scheduler.step()

    with pytest.warns(UserWarning):
        path = save_distributed_checkpoint(
            tmp_path / "complete-dcp", model, optimizer,
            scheduler=scheduler, scaler=scaler, metadata={"step": 3},
        )
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(3)
    optimizer.step()
    scheduler.step()
    random.random()
    np.random.random()
    torch.rand(3)

    with pytest.warns(UserWarning):
        metadata = load_distributed_checkpoint(
            path, model, optimizer, scheduler=scheduler, scaler=scaler,
        )

    assert scheduler.last_epoch == 1
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    torch.testing.assert_close(torch.rand(3), expected_torch)
    assert metadata["runtime_state_restored"]
    assert metadata["rng_restored"]
    assert metadata["checkpoint_world_size"] == 1
