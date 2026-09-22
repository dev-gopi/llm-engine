from evaluation.release_gate import GateRule, ReleaseGate


def test_release_gate_blocks_protected_regression():
    g=ReleaseGate([GateRule('accuracy','max',.01),GateRule('loss','min',.02)])
    assert not g.evaluate({'accuracy':.9,'loss':1.0},{'accuracy':.8,'loss':1.0})['passed']

def test_release_gate_allows_within_tolerance():
    g=ReleaseGate([GateRule('accuracy','max',.02)])
    assert g.evaluate({'accuracy':.9},{'accuracy':.89})['passed']
