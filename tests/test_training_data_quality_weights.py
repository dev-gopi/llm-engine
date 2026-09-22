import pytest

from training.data import _mixture_groups


def test_quality_weights_adjust_dataset_mixture_deterministically():
    config = {"dataset_weights": {"a": 0.5, "b": 0.5}, "dataset_quality_weights": {"a": 1.0, "b": 0.5}}
    groups = _mixture_groups(["data/a/train.jsonl", "data/b/train.jsonl"], [10, 10], config)
    assert groups == [(0, 10, 0.5), (10, 20, 0.25)]

def test_quality_weights_must_match_sources():
    with pytest.raises(ValueError):
        _mixture_groups(["data/a/train.jsonl"], [10], {"dataset_weights": {"a": 1}, "dataset_quality_weights": {"b": 1}})
