from evaluation.dynamic import anti_contamination_check, generate_math_fixture


def test_dynamic_fixture_reproducible():
    assert generate_math_fixture('v1',3)==generate_math_fixture('v1',3)
    assert anti_contamination_check([{'id':'v1-3'}],{'other'})['passed']
