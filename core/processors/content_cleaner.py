from __future__ import annotations

import re


class ContentCleaner:
    """Clean and normalize text content."""

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.replace("\x00", "")
        text = re.sub(r"\s+", " ", text)
        text = "".join(ch for ch in text if ord(ch) >= 32 or ch in "\n\r\t")
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\r", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
