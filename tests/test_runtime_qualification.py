from runtime.qualification import QualificationProbe, qualify_profile


def _probe(name: str, passed: bool, observed: str) -> QualificationProbe:
    return QualificationProbe(name, passed, observed, "test requirement")


def test_profile_qualification_is_evidence_driven():
    probes = [
        _probe("cuda", True, "available=True"),
        _probe("multi_node", True, "8"),
    ]
    result = qualify_profile(
        {
            "profile": "test",
            "qualification": {
                "requires_cuda": True,
                "requires_multi_node": True,
                "required_nodes": 8,
            },
        },
        probes,
    )
    assert result["qualified"] is True
    assert result["failures"] == []


def test_profile_qualification_reports_missing_cluster_requirements():
    probes = [
        _probe("cuda", False, "available=False"),
        _probe("multi_node", False, "1"),
    ]
    result = qualify_profile(
        {
            "profile": "large",
            "qualification": {
                "requires_cuda": True,
                "requires_multi_node": True,
                "required_nodes": 8,
            },
        },
        probes,
    )
    assert result["qualified"] is False
    assert any("CUDA" in item for item in result["failures"])
    assert any("8 nodes" in item for item in result["failures"])
