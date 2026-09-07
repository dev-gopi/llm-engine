import pytest

from scripts.prepare_hf_image_dataset import image_bytes


def test_hf_image_bytes_accepts_arrow_image_struct() -> None:
    assert image_bytes({"bytes": b"png", "path": None}) == b"png"
    assert image_bytes(b"jpeg") == b"jpeg"


def test_hf_image_bytes_rejects_missing_encoded_data() -> None:
    with pytest.raises(ValueError, match="encoded bytes"):
        image_bytes({"bytes": None, "path": "remote.png"})


def test_prepare_split_bounds_batches_and_output(tmp_path, monkeypatch) -> None:
    import io
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    from scripts import prepare_hf_image_dataset as module

    encoded = io.BytesIO()
    Image.new("RGB", (4, 4)).save(encoded, format="PNG")
    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({"image": [encoded.getvalue()] * 7,
                             "label": [0, 1, 0, 1, 0, 1, 0]}), source)
    monkeypatch.setattr(module, "parquet_urls", lambda *args: [source.as_uri()])
    original = pq.ParquetFile
    sizes = []

    class TrackedParquet:
        def __init__(self, path):
            self.reader = original(path)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.reader.close()

        def iter_batches(self, **kwargs):
            sizes.append(kwargs["batch_size"])
            yield from self.reader.iter_batches(**kwargs)

    monkeypatch.setattr(module.pq, "ParquetFile", TrackedParquet)
    output = tmp_path / "images"
    assert module.prepare_split("test", "default", "train", output,
                                image_column="image", label_column="label",
                                labels=("a", "b"), limit=3, timeout=1,
                                batch_size=2) == 3
    assert sizes == [2]
    assert len(list(output.rglob("*.png"))) == 3
    with Image.open(output / "a" / "000002.png") as image:
        assert image.mode == "RGB"


@pytest.mark.parametrize("options,match", [
    ({"batch_size": 0}, "batch_size"),
    ({"labels": ("../escape", "b")}, "safe directory"),
])
def test_prepare_split_rejects_invalid_options_before_io(tmp_path, options, match):
    from scripts.prepare_hf_image_dataset import prepare_split
    kwargs = dict(image_column="image", label_column="label", labels=("a", "b"),
                  limit=1, timeout=1)
    kwargs.update(options)
    with pytest.raises(ValueError, match=match):
        prepare_split("unused", "default", "train", tmp_path, **kwargs)
