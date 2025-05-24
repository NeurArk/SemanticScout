#!/usr/bin/env python
"""Environment setup script."""
from pathlib import Path


def create_directories() -> None:
    dirs = [
        Path("data/uploads"),
        Path("data/chroma_db"),
        Path("data/cache"),
        Path("logs"),
    ]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def main() -> None:
    create_directories()
    print("Environment setup complete.")


if __name__ == "__main__":
    main()
