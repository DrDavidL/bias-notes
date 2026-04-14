"""Fail fast if any notebook still contains outputs, execution counts, or local paths."""

from __future__ import annotations

import argparse
from pathlib import Path

from notebook_hygiene import format_issues, inspect_repo_notebooks, iter_notebook_paths
from project_config import PROJECT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(PROJECT_ROOT),
        help="Repo root to scan. Defaults to the current project root.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    notebooks = iter_notebook_paths(root)
    issues = inspect_repo_notebooks(root)
    if issues:
        print(format_issues(issues, root=root))
        return 1

    print(
        f"Notebook hygiene check passed for {len(notebooks)} notebook(s): "
        "no outputs, execution counts, or local path references found."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
