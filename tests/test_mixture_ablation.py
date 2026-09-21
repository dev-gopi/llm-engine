import pytest
from datasets.mixture_ablation import MixtureObservation, compare_mixtures

def test_compare_mixtures_is_descriptive_and_deterministic():
    result = compare_mixtures([
        MixtureObservation("base", {"web": 0.7, "code": 0.3}, {"math": 0.5, "code": 0.4}),
        MixtureObservation("code_heavy", {"web": 0.4, "code": 0.6}, {"math": 0.55, "code": 0.45}),
    ], "base")
    assert result["comparisons"][1]["metric_delta_vs_baseline"] == {"code": 0.05, "math": 0.05}
    assert "ranking" not in result

def test_mixture_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        MixtureObservation("bad", {"a": 0.2, "b": 0.2}, {"x": 1})
