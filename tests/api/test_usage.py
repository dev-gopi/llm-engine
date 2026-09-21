import pytest
from api.usage import account_usage

def test_usage_accounting_is_consistent():
    usage=account_usage(100,40,cached_tokens=25,reasoning_tokens=10,prefix_cache_hit=True)
    assert usage.total_tokens==140
    assert usage.as_dict()["cached_tokens"]==25
    assert usage.as_dict()["reasoning_tokens"]==10
    assert usage.as_dict()["cache"]["prefix_cache_hit"] is True

def test_usage_accounting_rejects_impossible_counts():
    with pytest.raises(ValueError): account_usage(10,5,cached_tokens=11)
    with pytest.raises(ValueError): account_usage(10,5,reasoning_tokens=6)
