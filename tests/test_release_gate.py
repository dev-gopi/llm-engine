from evaluation.release_gate import GateRule, ReleaseGate


def test_release_gate_blocks_protected_regression():
    g = ReleaseGate([GateRule("accuracy", "max", 0.01), GateRule("loss", "min", 0.02)])
    assert not g.evaluate(
        {"accuracy": 0.9, "loss": 1.0}, {"accuracy": 0.8, "loss": 1.0}
    )["passed"]


def test_release_gate_allows_within_tolerance():
    g = ReleaseGate([GateRule("accuracy", "max", 0.02)])
    assert g.evaluate({"accuracy": 0.9}, {"accuracy": 0.89})["passed"]
