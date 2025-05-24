#!/usr/bin/env python
"""Basic health check script."""
from __future__ import annotations

import json
from pathlib import Path

import requests

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.settings import get_settings


def check_openai() -> bool:
    settings = get_settings()
    url = "https://api.openai.com/v1/models"
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def check_directories() -> bool:
    settings = get_settings()
    dirs = [settings.chroma_persist_dir, settings.upload_dir, Path("logs")]
    return all(d.exists() for d in dirs)


def main() -> None:
    status = {
        "openai": check_openai(),
        "directories": check_directories(),
    }
    print(json.dumps(status))


if __name__ == "__main__":
    main()
