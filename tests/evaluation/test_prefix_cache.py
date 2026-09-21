from evaluation.prefix_cache import collect_prefix_cache_metrics


def test_prefix_cache_metrics_are_quantitative():
    cache=type("Cache",(),{"memory_bytes":1234,"evictions":2})()
    generator=type("Generator",(),{"prefix_cache":cache,"prefix_cache_hits":3,"prefix_cache_misses":4,"prefix_cache_tokens":90,"prefix_prefill_tokens_saved":90})()
    metrics=collect_prefix_cache_metrics(generator)
    assert metrics.hits == 3
    assert metrics.misses == 4
    assert metrics.cached_tokens == 90
    assert metrics.prefill_tokens_saved == 90
    assert metrics.memory_bytes == 1234
    assert metrics.evictions == 2
