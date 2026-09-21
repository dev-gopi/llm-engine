from evaluation.dynamic import generate_math_fixture, anti_contamination_check

def test_dynamic_fixture_reproducible():
    assert generate_math_fixture('v1',3)==generate_math_fixture('v1',3)
    assert anti_contamination_check([{'id':'v1-3'}],{'other'})['passed']
