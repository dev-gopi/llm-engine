from tests.asgi_client import ASGIClient

from serving.api import ServingSettings, create_app
from tests.test_serving import FakeBackend


def test_model_capabilities_are_advertised_conservatively():
    settings = ServingSettings(
        model_name="gopi-test",
        bot_name="Gopi",
        allowed_hosts=("testserver", "test", "localhost", "127.0.0.1"),
    )
    with ASGIClient(create_app(FakeBackend(), settings=settings)) as client:
        response = client.get("/v1/models")
        assert response.status_code == 200
        model = response.json()["data"][0]
        capabilities = model["capabilities"]
        assert capabilities["chat"] is True
        assert capabilities["streaming"] is True
        assert capabilities["structured_outputs"] is True
        assert capabilities["vision"] is False
        assert capabilities["audio"] is False

        response = client.get("/v1/models/gopi-test/capabilities")
        assert response.status_code == 200
        assert response.json()["capabilities"] == capabilities


def test_unknown_model_capabilities_is_404():
    settings = ServingSettings(
        model_name="gopi-test",
        bot_name="Gopi",
        allowed_hosts=("testserver", "test", "localhost", "127.0.0.1"),
    )
    with ASGIClient(create_app(FakeBackend(), settings=settings)) as client:
        assert client.get("/v1/models/nope/capabilities").status_code == 404
