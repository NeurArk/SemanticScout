from __future__ import annotations

from typing import List, Optional, Tuple
import logging

from core.chat_engine import ChatEngine
from core.embedder import EmbeddingService
from core.vector_store import VectorStore
from core.models.chat import ChatMessage
from core.models.search import SearchQuery

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Simple RAG pipeline combining search and chat."""

    def __init__(self) -> None:
        self.chat_engine = ChatEngine()
        self.vector_store = VectorStore()
        self.embedder = EmbeddingService()

    def query(
        self, question: str, history: Optional[List[ChatMessage]] = None
    ) -> Tuple[str, List[str]]:
        """Answer a question using document retrieval and GPT-4."""

        embedding = self.embedder.embed_query(question)
        search_query = SearchQuery(query_text=question, max_results=5)
        try:
            search_response = self.vector_store.search(embedding, search_query)
        except Exception as exc:  # pragma: no cover - wrapper
            logger.error("Vector search failed: %s", exc)
            return (
                "I couldn't find any relevant information in the documents.",
                [],
            )

        results = search_response.results
        if not results:
            return (
                "I couldn't find any relevant information in the documents.",
                [],
            )

        chunks = [res.content for res in results]
        sources = list({res.metadata.get("filename", "Unknown") for res in results})

        answer = self.chat_engine.chat(question, chunks, history)

        if sources and not any(src in answer for src in sources):
            answer += f"\n\nSources: {', '.join(sources)}"

        return answer, sources
