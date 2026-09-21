import json

from runtime.capabilities import discover_capabilities, load_capability_evidence
from tests.test_serving import FakeBackend


def test_capability_evidence_is_explicit(tmp_path):
    path=tmp_path/"evidence.json"
    path.write_text(json.dumps({"chat":True,"structured_outputs":False,"bad":"ignore"}))
    assert load_capability_evidence(path) == {"chat":True,"structured_outputs":False}


def test_unvalidated_client_capabilities_are_gated():
    capabilities=discover_capabilities(FakeBackend(), validation_evidence={"chat":True,"streaming":True,"structured_outputs":False})
    assert capabilities.chat is True
    assert capabilities.streaming is True
    assert capabilities.structured_outputs is False
