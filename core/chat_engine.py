from __future__ import annotations

from typing import List, Optional
import logging

import openai

from core.models.chat import ChatMessage
from config.settings import get_settings

logger = logging.getLogger(__name__)


class ChatEngine:
    """Simple chat engine using GPT-4."""

    def __init__(self) -> None:
        settings = get_settings()
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    def chat(
        self,
        query: str,
        context_chunks: List[str],
        history: Optional[List[ChatMessage]] = None,
    ) -> str:
        """Generate a chat response using provided document context."""

        context = "\n\n".join(
            [f"[Document excerpt {i + 1}]:\n{chunk}" for i, chunk in enumerate(context_chunks[:5])]
        )

        system_msg = (
            "You are a helpful assistant that answers questions based on provided documents.\n"
            "When answering, mention which document excerpt you're using.\n"
            "If the documents don't contain the answer, say so clearly."
        )
        messages = [{"role": "system", "content": system_msg}]

        if history:
            for msg in history[-3:]:
                messages.append({"role": msg.role.value, "content": msg.content})

        user_msg = f"Documents:\n{context}\n\nQuestion: {query}"
        messages.append({"role": "user", "content": user_msg})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as exc:  # pragma: no cover - simple wrapper
            logger.error("Chat generation failed: %s", exc)
            return f"Sorry, I encountered an error: {str(exc)}"
