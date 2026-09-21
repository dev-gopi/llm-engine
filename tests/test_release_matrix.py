from pathlib import Path

import yaml


def test_release_matrix_covers_all_required_capabilities_with_existing_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    matrix = yaml.safe_load((root / "configs/evaluation.release_matrix.yaml").read_text())

    assert matrix["schema_version"] == 1
    categories = matrix["required_categories"]
    assert set(categories) == {
        "knowledge", "math", "code", "reasoning", "instruction_following",
        "structured_json", "tools", "rag", "long_context", "safety",
        "hallucination", "refusal",
    }
    for category, gate in categories.items():
        assert (root / gate["artifact"]).is_file(), category
        assert gate["command"].startswith(".venv/bin/pytest tests/"), category
