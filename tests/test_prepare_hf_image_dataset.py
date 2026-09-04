import pytest

from scripts.prepare_hf_image_dataset import image_bytes


def test_hf_image_bytes_accepts_arrow_image_struct() -> None:
    assert image_bytes({"bytes": b"png", "path": None}) == b"png"
    assert image_bytes(b"jpeg") == b"jpeg"


def test_hf_image_bytes_rejects_missing_encoded_data() -> None:
    with pytest.raises(ValueError, match="encoded bytes"):
        image_bytes({"bytes": None, "path": "remote.png"})
