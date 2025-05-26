from __future__ import annotations

import os
from typing import Tuple

try:  # pragma: no cover - handled in tests
    import magic  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    magic = None

ALLOWED_EXTENSIONS = ["pdf", "docx", "txt", "md"]
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB


def validate_file_type(file_path: str) -> Tuple[bool, str]:
    """Validate file type using magic numbers."""
    if not os.path.exists(file_path):
        return False, "File not found"

    ext = os.path.splitext(file_path)[1].lower().lstrip(".")
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"File type '{ext}' not supported"

    if magic is None:
        return False, "libmagic not available"

    mime = magic.from_file(file_path, mime=True)
    valid_mimes = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "txt": "text/plain",
        "md": "text/plain",
    }

    expected_mime = valid_mimes.get(ext)
    if mime != expected_mime and not (ext in ["txt", "md"] and mime.startswith("text/")):
        return False, "File content doesn't match extension"

    return True, "Valid"


def validate_file_size(file_path: str) -> Tuple[bool, str]:
    """Validate file size."""
    size = os.path.getsize(file_path)
    if size > MAX_FILE_SIZE:
        return False, f"File size {size} exceeds maximum {MAX_FILE_SIZE}"
    return True, "Valid"


def sanitize_text(text: str) -> str:
    """Clean and sanitize text content."""
    text = text.replace("\x00", "")
    text = " ".join(text.split())
    return text.strip()


__all__ = ["validate_file_type", "validate_file_size", "sanitize_text"]
