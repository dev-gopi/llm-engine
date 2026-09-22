from evaluation.soak import evaluate_soak


def test_soak_passes_stable_run():
    r=evaluate_soak([{'ok':True,'memory_mb':100},{'ok':True,'memory_mb':105}])
    assert r['passed']
