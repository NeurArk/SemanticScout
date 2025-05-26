from __future__ import annotations

from typing import List
import logging

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from core.exceptions import EmbeddingError, RateLimitError
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class OpenAIEmbedder:
    """OpenAI embedding generation with retry logic."""

    def __init__(self) -> None:
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
        self.dimension = settings.embedding_dimension

    @retry(stop=stop_after_attempt(settings.max_retries), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimension,
            )
            return response.data[0].embedding
        except openai.RateLimitError as exc:
            logger.warning("Rate limit hit: %s", exc)
            raise RateLimitError(retry_after=settings.rate_limit_delay) from exc
        except openai.APIError as exc:
            logger.error("OpenAI API error: %s", exc)
            raise EmbeddingError(f"Failed to generate embedding: {exc}") from exc

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimension,
            )
            return [item.embedding for item in response.data]
        except openai.RateLimitError as exc:
            logger.warning("Rate limit hit in batch: %s", exc)
            embeddings = []
            for text in texts:
                embeddings.append(self.generate_embedding(text))
            return embeddings
        except openai.APIError as exc:
            logger.error("OpenAI API error during batch: %s", exc)
            raise EmbeddingError(f"Failed to generate embeddings batch: {exc}") from exc
