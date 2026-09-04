from pathlib import Path

import pytest
import torch
from PIL import Image

from image_data.audit import audit_images
from image_data.dataset import ImageClassificationDataset, ImageDataset
from image_data.processor import ImageProcessor, load_image, tensor_to_image


def save_image(path: Path, size: tuple[int, int] = (20, 10), color=(255, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_processor_resize_normalization_and_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "wide.png"
    save_image(source)
    legacy = load_image(source, 8)
    assert legacy.shape == (3, 8, 8)
    assert legacy.min() >= -1 and legacy.max() <= 1
    center = ImageProcessor(8, resize_mode="center_crop", normalization="zero_one")(source)
    assert center.shape == (3, 8, 8)
    assert center.min() >= 0 and center.max() <= 1
    assert tensor_to_image(legacy).size == (8, 8)


def test_processor_config_and_validation() -> None:
    processor = ImageProcessor.from_config({
        "image_size": 16, "image_normalization": "imagenet",
        "horizontal_flip": False, "color_jitter": 0.2,
    }, training=True)
    assert processor.resize_mode == "random_crop"
    assert processor.horizontal_flip_probability == 0
    with pytest.raises(ValueError, match="resize_mode"):
        ImageProcessor(8, resize_mode="invalid")


def test_datasets_share_custom_processor(tmp_path: Path) -> None:
    save_image(tmp_path / "a" / "one.png")
    save_image(tmp_path / "b" / "two.png", color=(0, 255, 0))
    processor = ImageProcessor(12, normalization="zero_one")
    plain = ImageDataset(tmp_path, 12, processor=processor)
    classified = ImageClassificationDataset(tmp_path, 12, processor=processor)
    assert plain[0].shape == (3, 12, 12)
    assert classified.class_to_id == {"a": 0, "b": 1}
    assert classified[1][1] == 1


def test_image_audit_finds_corruption_and_duplicates(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    duplicate = tmp_path / "duplicate.png"
    save_image(first, (9, 7))
    duplicate.write_bytes(first.read_bytes())
    (tmp_path / "broken.png").write_bytes(b"not an image")
    report = audit_images(tmp_path)
    assert report.images == 3
    assert report.readable == 2
    assert report.corrupt == 1
    assert report.exact_duplicates == 1
    assert (report.min_width, report.min_height) == (9, 7)
