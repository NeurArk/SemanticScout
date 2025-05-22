"""Logging configuration utilities."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .settings import Settings


def setup_logging(settings: Optional[Settings] = None) -> None:
    """Configure application logging."""
    settings = settings or Settings()

    log_file = Path(settings.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ],
    )
