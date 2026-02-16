"""Utilities shared across Marimo playground notebooks."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_root(module_file: str | Path) -> Path:
    """Ensure the repository root is on ``sys.path`` and return it."""
    project_root = Path(module_file).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root

