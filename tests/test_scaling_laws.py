import pytest

from training.scaling_laws import ScalingObservation, summarize_scaling


def test_scaling_summary_reports_loss_and_shared_capability_trends() -> None:
    result = summarize_scaling([
        ScalingObservation(10, 100, 3.0, {"math": .1, "code": .2}),
        ScalingObservation(20, 100, 2.0, {"math": .3, "code": .4}),
    ])
    assert result["observations"] == 2
    assert result["validation_loss_log_compute_slope"] < 0
    assert result["capability_math_log_compute_slope"] > 0


def test_scaling_summary_requires_distinct_valid_observations() -> None:
    item = ScalingObservation(10, 100, 3.0, {})
    with pytest.raises(ValueError, match="at least"):
        summarize_scaling([item])
    with pytest.raises(ValueError, match="distinct"):
        summarize_scaling([item, item])
