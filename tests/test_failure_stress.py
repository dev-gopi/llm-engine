from inference.failure_stress import RequestLifecycle, stress_summary

def test_failure_cleans_resources():
    e=RequestLifecycle('x',resources=3); e.cancel(); assert stress_summary([e])['passed']
