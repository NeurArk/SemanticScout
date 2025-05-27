from __future__ import annotations

from unittest.mock import Mock, patch

from core.chat_engine import ChatEngine
from core.models.chat import ChatMessage, MessageRole


@patch("openai.OpenAI")
def test_chat_engine_basic(mock_openai: Mock) -> None:
    client = Mock()
    mock_openai.return_value = client
    completion = Mock()
    completion.choices = [Mock(message=Mock(content="Answer"))]
    client.chat.completions.create.return_value = completion

    engine = ChatEngine()
    response = engine.chat("What?", ["Some context"], [])

    assert response == "Answer"
    client.chat.completions.create.assert_called_once()


@patch("openai.OpenAI")
def test_chat_engine_history(mock_openai: Mock) -> None:
    client = Mock()
    mock_openai.return_value = client
    completion = Mock()
    completion.choices = [Mock(message=Mock(content="Ok"))]
    client.chat.completions.create.return_value = completion

    history = [
        ChatMessage(role=MessageRole.USER, content="Hello"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Hi"),
        ChatMessage(role=MessageRole.USER, content="Question"),
    ]

    engine = ChatEngine()
    engine.chat("Next", ["ctx"], history)

    args, kwargs = client.chat.completions.create.call_args
    messages = kwargs["messages"]
    # system + last 3 history + user
    assert len(messages) == 5
