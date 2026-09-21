import pytest
from jsonschema import ValidationError
from schema.structured_outputs import make_spec, validate_structured_output, StructuredSchemaError, StructuredOutputValidationError

def test_nested_arrays_and_required_fields_validate():
    spec=make_spec(name="order", schema={"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"],"additionalProperties":False}}},"required":["items"],"additionalProperties":False}, strict=True)
    assert validate_structured_output('{"items":[{"id":1}]}', spec)["items"][0]["id"] == 1

def test_additional_properties_are_rejected():
    spec=make_spec(name="x",schema={"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":False},strict=True)
    with pytest.raises(StructuredOutputValidationError): validate_structured_output('{"a":"ok","b":1}',spec)

def test_malformed_schema_is_rejected():
    with pytest.raises(StructuredSchemaError): make_spec(name="x",schema={"type":"wat"})

def test_malformed_generation_is_rejected():
    spec=make_spec(name="x",schema={"type":"object"})
    with pytest.raises(StructuredOutputValidationError): validate_structured_output('{bad',spec)
from tests.asgi_client import ASGIClient
from serving.api import create_app, ServingSettings
from serving.runtime import BackendGeneration
from serving.schemas import FinishReason

class JsonBackend:
    ready=True
    def __init__(self,text): self.text=text
    async def generate(self,request): return BackendGeneration(self.text,3,2,FinishReason.STOP)
    async def stream(self,request):
        if False:
            yield None

def _settings(): return ServingSettings(model_name="gopi-test",bot_name="Gopi",allowed_hosts=("testserver","test","localhost","127.0.0.1"))

def test_chat_structured_success_is_schema_conforming():
    app=create_app(JsonBackend('{"a":"ok"}'),settings=_settings())
    with ASGIClient(app) as c:
        r=c.post('/v1/chat/completions',json={"model":"gopi-test","messages":[{"role":"user","content":"x"}],"response_format":{"type":"json_schema","json_schema":{"name":"x","strict":True,"schema":{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":False}}}})
    assert r.status_code==200 and r.json()["choices"][0]["message"]["content"]=='{"a":"ok"}'

def test_chat_structured_invalid_generation_is_incomplete():
    app=create_app(JsonBackend('{"a":1,"b":2}'),settings=_settings())
    with ASGIClient(app) as c:
        r=c.post('/v1/chat/completions',json={"model":"gopi-test","messages":[{"role":"user","content":"x"}],"response_format":{"type":"json_schema","json_schema":{"name":"x","strict":True,"schema":{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":False}}}})
    assert r.status_code==200 and r.json()["incomplete_details"]["reason"]=='structured_output_validation_failed'

def test_chat_structured_malformed_schema_is_rejected():
    app=create_app(JsonBackend('{}'),settings=_settings())
    with ASGIClient(app) as c:
        r=c.post('/v1/chat/completions',json={"model":"gopi-test","messages":[{"role":"user","content":"x"}],"response_format":{"type":"json_schema","json_schema":{"name":"x","strict":True,"schema":{"type":"wat"}}}})
    assert r.status_code==422
