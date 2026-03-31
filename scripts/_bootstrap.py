from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_root_on_path() -> Path:
    """
    Allow running scripts via: python scripts\\xx.py
    by adding the project root (parent of scripts/) to sys.path.
    """
    project_root = Path(__file__).resolve().parents[1]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return project_root

