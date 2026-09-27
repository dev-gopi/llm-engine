from serving.api import ServingSettings, create_app
from tests.asgi_client import ASGIClient
from tests.test_serving import FakeBackend


def settings():
    return ServingSettings(
        model_name="gopi-test",
        bot_name="Gopi",
        api_key="secret",
        allowed_hosts=("testserver", "test", "localhost", "127.0.0.1"),
    )


def test_embeddings_single_batch_and_metadata():
    with ASGIClient(create_app(FakeBackend(), settings=settings())) as client:
        headers = {"Authorization": "Bearer secret"}
        one = client.post(
            "/v1/embeddings",
            headers=headers,
            json={"model": "gopi-embedding-hash", "input": "hello world"},
        )
        batch = client.post(
            "/v1/embeddings",
            headers=headers,
            json={"model": "gopi-embedding-hash", "input": ["hello", "world"]},
        )
        meta = client.get("/v1/embeddings/models", headers=headers)
        assert one.status_code == batch.status_code == 200
        assert len(one.json()["data"][0]["embedding"]) == 384
        assert len(batch.json()["data"]) == 2
        assert batch.json()["usage"]["total_tokens"] == 2
        assert meta.json()["data"][0]["embedding_dimension"] == 384


def test_embeddings_reject_unknown_model_empty_input_and_too_many_dimensions():
    with ASGIClient(create_app(FakeBackend(), settings=settings())) as client:
        headers = {"Authorization": "Bearer secret"}
        assert (
            client.post(
                "/v1/embeddings", headers=headers, json={"model": "wrong", "input": "x"}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/v1/embeddings",
                headers=headers,
                json={"model": "gopi-embedding-hash", "input": []},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/v1/embeddings",
                headers=headers,
                json={"model": "gopi-embedding-hash", "input": "x", "dimensions": 385},
            ).status_code
            == 422
        )


def test_embeddings_support_reduced_dimensions_and_base64():
    import base64
    import struct

    with ASGIClient(create_app(FakeBackend(), settings=settings())) as client:
        headers = {"Authorization": "Bearer secret"}
        reduced = client.post(
            "/v1/embeddings",
            headers=headers,
            json={"model": "gopi-embedding-hash", "input": "hello", "dimensions": 128},
        )
        encoded = client.post(
            "/v1/embeddings",
            headers=headers,
            json={
                "model": "gopi-embedding-hash",
                "input": "hello",
                "dimensions": 8,
                "encoding_format": "base64",
            },
        )
        assert reduced.status_code == encoded.status_code == 200
        assert len(reduced.json()["data"][0]["embedding"]) == 128
        raw = base64.b64decode(encoded.json()["data"][0]["embedding"])
        assert len(raw) == 8 * 4
        assert len(struct.unpack("<8f", raw)) == 8
