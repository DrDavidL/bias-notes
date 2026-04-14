"""Configure Git to use the repo-managed hooks directory."""

from __future__ import annotations

import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    subprocess.run(
        ["git", "config", "core.hooksPath", ".githooks"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    print("Configured git core.hooksPath=.githooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
