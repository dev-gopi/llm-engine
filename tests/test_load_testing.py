from evaluation.load_testing import LoadSample, percentile, release_gate, summarize


def test_load_report_percentiles_and_release_gate():
    samples = [LoadSample(value, 200, completion_tokens=10) for value in (1, 2, 3, 4)]
    report = summarize(samples, 2)
    assert percentile([1, 2, 3, 4], .5) == 2.5
    assert report["requests_per_second"] == 2
    assert report["tokens_per_second"] == 20
    assert release_gate(report, max_p95_seconds=4, max_failure_rate=0)


def test_load_report_counts_failures_overload_and_cancellation():
    report = summarize([
        LoadSample(.1, 429), LoadSample(.2, 500, error="server"),
        LoadSample(.05, None, cancelled=True),
    ], 1)
    assert report["overloaded"] == 1
    assert report["failed"] == 1
    assert report["cancelled"] == 1
    assert not release_gate(report, max_p95_seconds=1, max_failure_rate=.1)
