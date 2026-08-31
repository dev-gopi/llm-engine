"""Sandboxed workspace operations for reviewed coding-agent workflows."""

from __future__ import annotations

import difflib
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable


IGNORED_DIRECTORIES = {".git", ".venv", "__pycache__", "checkpoints", "data", "exports"}
TEXT_SUFFIXES = {
    ".py", ".pyi", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".toml",
    ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".sh", ".sql", ".xml",
}


class WorkspaceService:
    def __init__(self, root: str | Path, *, timeout_seconds: float = 120.0) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"workspace root is not a directory: {self.root}")
        self.timeout_seconds = timeout_seconds

    def _path(self, relative: str) -> Path:
        if not relative or Path(relative).is_absolute():
            raise ValueError("workspace paths must be non-empty and relative")
        candidate = (self.root / relative).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError("workspace path escapes the configured root")
        return candidate

    @staticmethod
    def _hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def read(self, path: str, *, max_chars: int = 20_000) -> dict[str, Any]:
        source = self._path(path)
        if not source.is_file():
            raise FileNotFoundError(f"workspace file not found: {path}")
        if source.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("workspace read refuses files larger than 2MB")
        data = source.read_bytes()
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("workspace read supports UTF-8 text files only") from error
        return {
            "path": path, "content": content[:max_chars], "truncated": len(content) > max_chars,
            "sha256": self._hash(data), "size": len(data),
        }

    def search(self, query: str, *, limit: int = 50) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("search query cannot be empty")
        needle = query.casefold()
        matches = []
        for directory, names, files in os.walk(self.root):
            names[:] = [name for name in names if name not in IGNORED_DIRECTORIES]
            for name in files:
                path = Path(directory) / name
                if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 2 * 1024 * 1024:
                    continue
                try:
                    with path.open(encoding="utf-8") as stream:
                        for line_number, line in enumerate(stream, 1):
                            if needle in line.casefold():
                                matches.append({
                                    "path": str(path.relative_to(self.root)),
                                    "line": line_number,
                                    "text": line.rstrip()[:500],
                                })
                                if len(matches) >= limit:
                                    return {"query": query, "matches": matches, "truncated": True}
                except (OSError, UnicodeDecodeError):
                    continue
        return {"query": query, "matches": matches, "truncated": False}

    def edit(
        self, path: str, content: str, *, expected_sha256: str | None, apply: bool
    ) -> dict[str, Any]:
        destination = self._path(path)
        previous = destination.read_bytes() if destination.exists() else b""
        if destination.exists() and expected_sha256 != self._hash(previous):
            raise ValueError("expected_sha256 is required and must match the current file")
        new_data = content.encode("utf-8")
        diff = "".join(difflib.unified_diff(
            previous.decode("utf-8", errors="replace").splitlines(keepends=True),
            content.splitlines(keepends=True),
            fromfile=f"a/{path}", tofile=f"b/{path}",
        ))
        if apply:
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(new_data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_name, destination)
            finally:
                Path(temporary_name).unlink(missing_ok=True)
        return {
            "path": path, "applied": apply, "diff": diff[:50_000],
            "sha256": self._hash(new_data),
        }

    def apply_patch(self, patch: str, *, apply: bool) -> dict[str, Any]:
        if not patch.strip() or len(patch) > 262_144:
            raise ValueError("patch must contain 1 to 262144 characters")
        paths = []
        for line in patch.splitlines():
            if line.startswith(("--- ", "+++ ")):
                value = line[4:].split("\t", 1)[0]
                if value == "/dev/null":
                    raise ValueError("file creation/deletion patches are not supported")
                relative = value[2:] if value.startswith(("a/", "b/")) else value
                self._path(relative)
                paths.append(relative)
        if not paths:
            raise ValueError("patch contains no file paths")
        command = ["git", "apply", "--check", "--whitespace=error-all", "-"]
        checked = self._run(command, input_text=patch)
        if apply:
            self._run(["git", "apply", "--whitespace=error-all", "-"], input_text=patch)
        return {"paths": sorted(set(paths)), "applied": apply, "check": checked["stdout"] or "ok"}

    def run_test(self, preset: str) -> dict[str, Any]:
        commands = {
            "all": [str(self.root / ".venv/bin/python"), "-m", "pytest", "-q"],
            "unit": [str(self.root / ".venv/bin/python"), "-m", "pytest", "tests", "-q"],
        }
        if preset not in commands:
            raise ValueError(f"unknown test preset: {preset}")
        return self._run(commands[preset])

    def git(self, operation: str) -> dict[str, Any]:
        commands = {
            "status": ["git", "status", "--short"],
            "diff": ["git", "diff", "--no-ext-diff"],
            "diff_staged": ["git", "diff", "--cached", "--no-ext-diff"],
            "log": ["git", "log", "-20", "--oneline", "--decorate"],
        }
        if operation not in commands:
            raise ValueError(f"unknown read-only Git operation: {operation}")
        return self._run(commands[operation])

    def _run(self, command: list[str], *, input_text: str | None = None) -> dict[str, Any]:
        completed = subprocess.run(
            command, cwd=self.root, input=input_text, text=True, capture_output=True,
            timeout=self.timeout_seconds, check=False,
        )
        result = {
            "command": command, "returncode": completed.returncode,
            "stdout": completed.stdout[-100_000:], "stderr": completed.stderr[-100_000:],
        }
        if completed.returncode != 0:
            raise ValueError(f"command failed ({completed.returncode}): {completed.stderr[-2000:]}")
        return result

    def execute(self, actions: Iterable[Any]) -> list[dict[str, Any]]:
        results = []
        for action in actions:
            if action.type == "read":
                result = self.read(action.path)
            elif action.type == "search":
                result = self.search(action.query)
            elif action.type == "edit":
                result = self.edit(
                    action.path, action.content,
                    expected_sha256=action.expected_sha256, apply=action.apply,
                )
            elif action.type == "patch":
                result = self.apply_patch(action.content, apply=action.apply)
            elif action.type == "test":
                result = self.run_test(action.preset)
            elif action.type == "git":
                result = self.git(action.operation)
            else:
                raise ValueError(f"unsupported workspace action: {action.type}")
            results.append({"type": action.type, "result": result})
        return results
