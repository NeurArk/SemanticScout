from __future__ import annotations

from typing import List, Optional, Tuple
import logging

from core.chat_engine import ChatEngine
from core.embedder import EmbeddingService
from core.vector_store import VectorStore
from core.models.chat import ChatMessage
from core.models.search import SearchQuery
from core.utils.adaptive_search import adaptive_analyzer

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
        
        # Expand short queries for better retrieval
        expanded_question = adaptive_analyzer.expand_query(question)
        
        # Get initial embedding
        embedding = self.embedder.embed_query(expanded_question)
        
        # First pass with adaptive threshold
        initial_threshold = adaptive_analyzer.get_adaptive_threshold(question)
        
        search_query = SearchQuery(
            query_text=expanded_question, 
            max_results=10,  # Get more results for better selection
            similarity_threshold=initial_threshold
        )
        
        try:
            search_response = self.vector_store.search(embedding, search_query)
        except Exception as exc:  # pragma: no cover - wrapper
            logger.error("Vector search failed: %s", exc)
            return (
                "I couldn't find any relevant information in the documents.",
                [],
            )

        results = search_response.results
        
        # If no results, try with a lower threshold
        if not results and initial_threshold > 0.2:
            logger.info(f"No results with threshold {initial_threshold:.3f}, retrying with 0.15")
            fallback_query = SearchQuery(
                query_text=expanded_question,
                max_results=10,
                similarity_threshold=0.15
            )
            try:
                search_response = self.vector_store.search(embedding, fallback_query)
                results = search_response.results
            except Exception:
                pass
        
        if not results:
            return (
                "I couldn't find any relevant information in the documents.",
                [],
            )
        
        # Take top 5 results for context
        top_results = results[:5]
        chunks = [res.content for res in top_results]
        sources = list({res.metadata.get("filename", "Unknown") for res in top_results})
        
        # Log search effectiveness
        logger.info(
            f"Query: '{question}' (expanded: '{expanded_question}') - "
            f"Threshold: {initial_threshold:.3f} - "
            f"Found: {len(results)} results - "
            f"Using top {len(top_results)} for context"
        )

        answer = self.chat_engine.chat(question, chunks, history)

        if sources and not any(src in answer for src in sources):
            answer += f"\n\nSources: {', '.join(sources)}"

        return answer, sources
