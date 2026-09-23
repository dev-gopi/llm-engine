import pytest

from local_dataset.packing import batch_padding_efficiency, packing_efficiency


def test_packing_efficiency_reports_useful_and_waste():
    result = packing_efficiency([8, 8, 4], 8)
    assert result["useful_tokens"] == 20
    assert result["wasted_tokens"] == 4
    assert result["utilization"] == pytest.approx(20 / 24, abs=1e-8)

def test_batch_padding_accounts_for_padding_multiple():
    result = batch_padding_efficiency([5, 9, 3], batch_size=2, pad_to_multiple_of=8)
    assert result["allocated_tokens"] == 40
    assert result["padding_tokens"] == 23
    assert result["utilization"] == pytest.approx(17 / 40, abs=1e-8)
