from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .document import DocumentChunk


class MessageRole(str, Enum):
    """Role of a chat message."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """Represents a single chat message."""

    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatContext(BaseModel):
    """Context for generating chat responses."""

    messages: List[ChatMessage]
    retrieved_chunks: List[DocumentChunk] = Field(default_factory=list)
    max_context_length: int = Field(default=8000)

    def get_context_string(self) -> str:
        """Format retrieved chunks as context."""
        if not self.retrieved_chunks:
            return ""

        context_parts = []
        for chunk in self.retrieved_chunks:
            source = chunk.metadata.get("filename", "Unknown")
            context_parts.append(f"[Source: {source}]\n{chunk.content}\n")

        return "\n---\n".join(context_parts)

    def format_for_llm(self) -> List[Dict[str, str]]:
        """Format messages for OpenAI API."""
        formatted: List[Dict[str, str]] = []

        # Add system message with context if available
        context = self.get_context_string()
        if context:
            system_content = (
                "You are a helpful AI assistant with access to the user's documents.\n\n"
                "Based on the following document excerpts, answer the user's questions:\n\n"
                f"{context}\n\nIf the answer isn't in the provided context, say so."
            )
        else:
            system_content = (
                "You are a helpful AI assistant. The user hasn't uploaded any documents yet."
            )

        formatted.append({"role": "system", "content": system_content})

        # Add conversation history
        for msg in self.messages:
            formatted.append({"role": msg.role.value, "content": msg.content})

        return formatted


__all__ = ["ChatMessage", "ChatContext", "MessageRole"]
