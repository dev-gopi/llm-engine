import json

from scripts.download_rag_dataset import DATASET, SNAPSHOT, merged_manifest


def test_manifest_merge_preserves_existing_language_counts(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "dataset": DATASET,
        "snapshot": SNAPSHOT,
        "languages": {"bn": 10_000, "hi": 10_000},
    }), encoding="utf-8")
    manifest = merged_manifest(path, {"en": 20_000})
    assert manifest["languages"] == {"bn": 10_000, "hi": 10_000, "en": 20_000}
    assert manifest["license"] == ["CC-BY-SA-3.0", "GFDL"]
