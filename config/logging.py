import logging
from logging.config import dictConfig
from pathlib import Path
from .settings import get_settings


def setup_logging() -> None:
    """Configure application logging."""

    settings = get_settings()
    log_dir = Path(settings.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    logging_config = {
        "version": 1,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": settings.log_level,
            },
            "file": {
                "class": "logging.FileHandler",
                "formatter": "default",
                "filename": str(settings.log_file),
                "level": settings.log_level,
            },
        },
        "root": {
            "handlers": ["console", "file"],
            "level": settings.log_level,
        },
    }

    dictConfig(logging_config)


__all__ = ["setup_logging"]
