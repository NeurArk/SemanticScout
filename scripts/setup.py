#!/usr/bin/env python3
"""Initialize SemanticScout environment."""
from __future__ import annotations

from pathlib import Path

DIRECTORIES = [
    "data/uploads",
    "data/chroma_db",
    "data/cache",
    "data/logs",
    "logs",
]


def create_directories() -> None:
    """Create required directories with proper permissions."""
    for directory in DIRECTORIES:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o755)


def main() -> None:
    create_directories()
    print("Environment setup complete.")


if __name__ == "__main__":
    main()
