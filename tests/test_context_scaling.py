from evaluation.context_scaling import compare_context_runs


def test_context_scaling_requires_all_lengths():
    r=compare_context_runs([{'context_length':x,'quality_delta':0,'peak_memory_mb':x,'tokens_per_second':1} for x in (512,1024,2048,4096,8192)])
    assert r['complete']
