import hashlib
from types import SimpleNamespace

import pytest

from serving.workspace import WorkspaceService


def action(action_type, **values):
    defaults = {
        "path": "", "query": "", "content": "", "expected_sha256": None,
        "apply": False, "preset": "unit", "operation": "status",
    }
    defaults.update(values)
    return SimpleNamespace(type=action_type, **defaults)


def test_workspace_read_search_and_path_boundary(tmp_path):
    source = tmp_path / "src" / "example.py"
    source.parent.mkdir()
    source.write_text("answer = 42\n", encoding="utf-8")
    workspace = WorkspaceService(tmp_path)

    result = workspace.read("src/example.py")
    assert result["content"] == "answer = 42\n"
    assert result["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert workspace.search("ANSWER")["matches"][0]["line"] == 1
    with pytest.raises(ValueError, match="escapes"):
        workspace.read("../outside.txt")


def test_workspace_edit_requires_review_and_current_hash(tmp_path):
    source = tmp_path / "example.txt"
    source.write_text("old\n", encoding="utf-8")
    workspace = WorkspaceService(tmp_path)
    digest = workspace.read("example.txt")["sha256"]

    preview = workspace.edit(
        "example.txt", "new\n", expected_sha256=digest, apply=False
    )
    assert not preview["applied"]
    assert "-old" in preview["diff"] and "+new" in preview["diff"]
    assert source.read_text(encoding="utf-8") == "old\n"
    workspace.edit("example.txt", "new\n", expected_sha256=digest, apply=True)
    assert source.read_text(encoding="utf-8") == "new\n"
    with pytest.raises(ValueError, match="must match"):
        workspace.edit("example.txt", "stale\n", expected_sha256=digest, apply=True)


def test_workspace_patch_is_checked_before_apply(tmp_path):
    source = tmp_path / "example.txt"
    source.write_text("old\n", encoding="utf-8")
    workspace = WorkspaceService(tmp_path)
    patch = "--- a/example.txt\n+++ b/example.txt\n@@ -1 +1 @@\n-old\n+new\n"

    assert not workspace.apply_patch(patch, apply=False)["applied"]
    assert source.read_text(encoding="utf-8") == "old\n"
    workspace.apply_patch(patch, apply=True)
    assert source.read_text(encoding="utf-8") == "new\n"
    with pytest.raises(ValueError, match="creation/deletion"):
        workspace.apply_patch("--- /dev/null\n+++ b/new.txt\n", apply=False)


def test_workspace_only_exposes_allowlisted_commands(tmp_path, monkeypatch):
    workspace = WorkspaceService(tmp_path)
    calls = []
    monkeypatch.setattr(
        workspace, "_run", lambda command, **kwargs: calls.append(command) or {"returncode": 0}
    )

    workspace.run_test("unit")
    workspace.git("status")
    assert calls[0][-4:] == ["-m", "pytest", "tests", "-q"]
    assert calls[1] == ["git", "status", "--short"]
    with pytest.raises(ValueError, match="unknown test"):
        workspace.run_test("shell")


def test_workspace_executes_a_bounded_sequence(tmp_path):
    (tmp_path / "a.txt").write_text("needle\n", encoding="utf-8")
    results = WorkspaceService(tmp_path).execute([
        action("read", path="a.txt"), action("search", query="needle")
    ])
    assert [item["type"] for item in results] == ["read", "search"]
