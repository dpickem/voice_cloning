#!/usr/bin/env python3
"""Run all unit and integration tests for the repository."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


def _existing_dirs(paths: Iterable[Path]) -> List[Path]:
    """Return only the paths that exist."""
    return [path for path in paths if path.exists()]


def _run_pytest(directories: List[Path], label: str, repo_root: Path) -> bool:
    """
    Run pytest for the provided directories.

    Returns True if the tests succeed or there is nothing to run; False otherwise.
    """
    if not directories:
        print(f"[skip] No {label} test directories found.")
        return True

    cmd = ["python", "-m", "pytest", *[str(p) for p in directories]]
    print(f"[run] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=repo_root)
    if result.returncode != 0:
        print(f"[fail] {label} tests failed (exit {result.returncode}).")
        return False

    print(f"[ok] {label} tests passed.")
    return True


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    unit_candidates = [
        repo_root / "server" / "tests",
        repo_root / "tests",
    ]
    integration_candidates = [
        repo_root / "server" / "integration_tests",
        repo_root / "integration_tests",
        repo_root / "tests" / "integration",
    ]

    unit_dirs = _existing_dirs(unit_candidates)
    integration_dirs = _existing_dirs(integration_candidates)

    unit_ok = _run_pytest(unit_dirs, "unit", repo_root)
    integration_ok = _run_pytest(integration_dirs, "integration", repo_root)

    if unit_ok and integration_ok:
        sys.exit(0)

    sys.exit(1)


if __name__ == "__main__":
    main()
