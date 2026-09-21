from pathlib import Path

from scripts.audit_task_registry import audit_task_registry


def test_repository_task_registry_has_unique_resolved_ids() -> None:
    report = audit_task_registry(Path(__file__).resolve().parents[1] / "docs" / "TASKS.md")
    assert report["passed"], report
    assert report["sections"] == report["unique_ids"]


def test_registry_audit_detects_duplicates_and_missing_dependencies(tmp_path) -> None:
    registry = tmp_path / "TASKS.md"
    registry.write_text(
        "### A-001: One\n\n- **ID**: `A-001`\n\n- **Dependencies**: `MISSING-001`\n\n"
        "### A-001: Two\n\n- **ID**: `A-002`\n",
        encoding="utf-8",
    )
    report = audit_task_registry(registry)
    assert not report["passed"]
    assert report["duplicate_ids"] == ["A-001"]
    assert report["mismatched_ids"] == [{"heading": "A-001", "declared": "A-002"}]
    assert report["undefined_dependencies"] == ["MISSING-001"]
