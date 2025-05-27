from __future__ import annotations

# pragma: no cover

from pathlib import Path
from typing import Optional

from ..exceptions import SemanticScoutError


def safe_read(path: str) -> str:
    """Safely read a file and return its contents."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - trivial
        raise SemanticScoutError("Failed to read file", {"path": path}) from exc


def safe_write(path: str, data: str) -> None:
    """Safely write data to a file."""
    try:
        Path(path).write_text(data, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - trivial
        raise SemanticScoutError("Failed to write file", {"path": path}) from exc


__all__ = ["safe_read", "safe_write"]
