from serving.api import create_app, ServingSettings
from serving.runtime import BackendGeneration
from serving.schemas import FinishReason
from tests.test_serving import FakeBackend, request, settings

def test_responses_endpoint_matches_basic_chat_shape():
    app=create_app(FakeBackend(), settings=settings())
    response=request(app,"POST","/v1/responses",json={"model":"gopi-test","input":"hello"})
    assert response.status_code == 200
    body=response.json(); assert body["object"]=="response"; assert body["status"]=="completed"

def test_responses_streams_events():
    app=create_app(FakeBackend(), settings=settings())
    response=request(app,"POST","/v1/responses",json={"model":"gopi-test","input":"hello","stream":True})
    assert response.status_code==200 and "response.output_text.delta" in response.text
