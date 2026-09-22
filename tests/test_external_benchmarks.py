from evaluation.external import BenchmarkAdapter, BenchmarkSpec


def test_versioned_adapter():
    spec=BenchmarkSpec('toy','1','abc','accuracy'); r=BenchmarkAdapter(spec,lambda a,b: a==b).run([(1,1),(1,2)],controls={'seed':1})
    assert r.count==2 and r.score==.5 and r.spec.fingerprint()
