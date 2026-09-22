import asyncio
import base64

import pytest

from serving.vision_runtime import image_url_to_bytes, message_image_urls


def test_message_image_urls_extracts_only_openai_image_parts():
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,YQ=="}},
        ],
    }]
    assert message_image_urls(messages) == ["data:image/png;base64,YQ=="]


def test_data_url_decode_and_size_limit():
    payload = b"test-image-bytes"
    encoded = base64.b64encode(payload).decode("ascii")
    url = f"data:image/png;base64,{encoded}"
    assert asyncio.run(image_url_to_bytes(url, allow_remote=False, max_bytes=1024)) == payload
    with pytest.raises(ValueError, match="serving limit"):
        asyncio.run(image_url_to_bytes(url, allow_remote=False, max_bytes=2))


def test_remote_images_are_disabled_by_default():
    with pytest.raises(ValueError, match="GOPI_VISION_ALLOW_REMOTE_IMAGES"):
        asyncio.run(image_url_to_bytes(
            "https://example.com/image.png",
            allow_remote=False,
            max_bytes=1024,
        ))
