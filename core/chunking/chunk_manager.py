from __future__ import annotations

from typing import List

from core.models.document import DocumentChunk


class ChunkManager:
    """Utility to manage overlapping chunks."""

    def __init__(self, overlap: int = 200) -> None:
        self.overlap = overlap

    def apply_overlap(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        if not chunks:
            return []
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            curr = chunks[i]
            if prev.end_char - self.overlap < curr.start_char:
                # adjust start to include overlap
                curr.start_char = max(prev.end_char - self.overlap, 0)
        return chunks
