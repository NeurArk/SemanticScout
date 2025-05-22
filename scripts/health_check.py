#!/usr/bin/env python3
"""Simple health check for SemanticScout environment."""
from __future__ import annotations

from pathlib import Path
from typing import Dict
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from config.settings import Settings


def check_directories(settings: Settings) -> Dict[str, bool]:
    """Verify critical directories exist."""
    return {
        "uploads": Path(settings.upload_dir).exists(),
        "chroma_db": Path(settings.chroma_persist_dir).exists(),
        "logs": Path(settings.log_file).parent.exists(),
    }


def main() -> None:
    env_file = Path(".env") if Path(".env").exists() else Path(".env.example")
    settings = Settings(_env_file=env_file)
    results = check_directories(settings)
    for key, status in results.items():
        print(f"{key}: {'OK' if status else 'MISSING'}")


if __name__ == "__main__":
    main()
