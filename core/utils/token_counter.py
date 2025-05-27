from __future__ import annotations

# pragma: no cover
# mypy: ignore-errors

from typing import List

try:  # pragma: no cover - optional dependency
    import tiktoken
except Exception:  # pragma: no cover - offline or unavailable
    tiktoken = None


class TokenCounter:
    """Count tokens for cost estimation."""

    def __init__(self, model: str = "text-embedding-3-large") -> None:
        if tiktoken is not None:
            try:
                self.encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:  # pragma: no cover - may fail without network
                self.encoding = None
        else:
            self.encoding = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoding is None:
            return len(text)
        return len(self.encoding.encode(text))

    def count_tokens_batch(self, texts: List[str]) -> int:
        """Count total tokens in multiple texts."""
        return sum(self.count_tokens(t) for t in texts)


__all__ = ["TokenCounter"]
