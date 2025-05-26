from __future__ import annotations

import re
from typing import List, Tuple


def clean_text(text: str) -> str:
    """Clean text for processing."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.\,\!\?\-\:\;\'\"]+", "", text)
    return text.strip()


def extract_sentences(text: str) -> List[str]:
    """Extract sentences from text."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def calculate_overlap(text: str, start: int, end: int, overlap_size: int) -> Tuple[int, int]:
    """Calculate chunk boundaries with overlap."""
    new_start = max(0, start - overlap_size)
    new_end = min(len(text), end + overlap_size)
    return new_start, new_end


__all__ = ["clean_text", "extract_sentences", "calculate_overlap"]
