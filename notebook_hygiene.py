"""Helpers for verifying that committed notebooks are Git-safe."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


LOCAL_PATH_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"/Users/[^/\s]+"), "contains a macOS absolute path"),
    (re.compile(r"/home/[^/\s]+"), "contains a Linux absolute path"),
    (re.compile(r"[A-Za-z]:\\\\[^\\\s]+"), "contains a Windows absolute path"),
    (re.compile(r"file://"), "contains a file:// link"),
    (re.compile(r"OneDrive", re.IGNORECASE), "contains a OneDrive reference"),
)


@dataclass(frozen=True)
class NotebookIssue:
    path: Path
    kind: str
    message: str
    cell_index: int | None = None


def iter_notebook_paths(root: Path) -> list[Path]:
    notebooks: list[Path] = []
    for path in root.rglob("*.ipynb"):
        if ".ipynb_checkpoints" in path.parts:
            continue
        notebooks.append(path)
    return sorted(notebooks)


def _cell_source_text(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    return str(source)


def inspect_notebook(path: Path) -> list[NotebookIssue]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    issues: list[NotebookIssue] = []

    for cell_index, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("outputs"):
            issues.append(
                NotebookIssue(
                    path=path,
                    kind="outputs",
                    message="has saved cell outputs",
                    cell_index=cell_index,
                )
            )
        if cell.get("execution_count") is not None:
            issues.append(
                NotebookIssue(
                    path=path,
                    kind="execution_count",
                    message="has saved execution counts",
                    cell_index=cell_index,
                )
            )

        source_text = _cell_source_text(cell)
        for pattern, description in LOCAL_PATH_PATTERNS:
            if pattern.search(source_text):
                issues.append(
                    NotebookIssue(
                        path=path,
                        kind="local_path",
                        message=description,
                        cell_index=cell_index,
                    )
                )
                break

    raw_text = path.read_text(encoding="utf-8")
    for pattern, description in LOCAL_PATH_PATTERNS:
        if pattern.search(raw_text):
            issues.append(
                NotebookIssue(
                    path=path,
                    kind="local_path",
                    message=f"{description} in notebook metadata",
                )
            )
            break

    return issues


def inspect_repo_notebooks(root: Path) -> list[NotebookIssue]:
    issues: list[NotebookIssue] = []
    for notebook_path in iter_notebook_paths(root):
        issues.extend(inspect_notebook(notebook_path))
    return issues


def format_issues(issues: list[NotebookIssue], root: Path | None = None) -> str:
    if not issues:
        return "All notebooks are clean."

    lines = ["Notebook hygiene check failed:"]
    for issue in issues:
        display_path = issue.path
        if root is not None:
            try:
                display_path = issue.path.relative_to(root)
            except ValueError:
                display_path = issue.path
        suffix = f" (cell {issue.cell_index})" if issue.cell_index is not None else ""
        lines.append(f"- {display_path}{suffix}: {issue.message}")
    return "\n".join(lines)
