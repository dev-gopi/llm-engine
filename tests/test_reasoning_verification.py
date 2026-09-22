from reasoning.curriculum import order_curriculum
from reasoning.verification import verify_code, verify_logic, verify_math


def test_math_and_code_verification():
    assert verify_math('steps... 42',42).passed
    assert verify_code('ok','ok').passed
    assert verify_logic('YES','yes').passed

def test_curriculum_deterministic():
    x=[{'id':'b','difficulty':.8},{'id':'a','difficulty':.2}]
    assert [i.id for i in order_curriculum(x)]==['a','b']
