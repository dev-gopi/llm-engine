import pytest

from inference.tensor_parallel import validate_tensor_parallel_size


def test_tensor_parallel_topology_validation_is_explicit(monkeypatch) -> None:
    validate_tensor_parallel_size(1, attention_heads=8, kv_heads=2)
    with pytest.raises(ValueError, match="divisible"):
        validate_tensor_parallel_size(3, attention_heads=8, kv_heads=2)
    monkeypatch.setenv("WORLD_SIZE", "2")
    validate_tensor_parallel_size(2, attention_heads=8, kv_heads=2)
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(RuntimeError, match="torchrun world size"):
        validate_tensor_parallel_size(2, attention_heads=8, kv_heads=2)
