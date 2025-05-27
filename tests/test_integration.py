import pytest
from unittest.mock import patch, Mock

import app


def test_gradio_functions():
    with patch.object(app.vector_store, "clear", return_value=None) as clear_mock:
        result = app.clear_all_documents()
        assert "cleared" in result.lower()
        clear_mock.assert_called_once()

    with patch.object(app.rag_pipeline, "query", return_value=("Hello", [])) as qmock:
        response = app.chat_response("Hi", [])
        assert len(response) > 0
        assert "error" not in response.lower()
        qmock.assert_called_once()
