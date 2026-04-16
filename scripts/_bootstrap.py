from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_root_on_path() -> Path:
    """Locate the repo root from any nested script path and add it to sys.path."""
    current = Path(__file__).resolve()
    for candidate in [current.parent, *current.parents]:
        if (candidate / "src").exists():
            root_str = str(candidate)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            return candidate
    raise RuntimeError("Could not locate project root.")
