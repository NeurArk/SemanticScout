import os
import pytest

from core.utils.validation import sanitize_text, validate_file_size, validate_file_type


def test_sanitize_text() -> None:
    """Sanitize text removes null bytes and extra spaces."""
    text = "Hello\x00\nWorld"
    assert sanitize_text(text) == "Hello World"


def test_validate_file_size(tmp_path) -> None:  # type: ignore[no-untyped-def]
    file_path = tmp_path / "file.txt"
    file_path.write_text("data")
    valid, _ = validate_file_size(str(file_path))
    assert valid is True


def test_validate_file_type(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("magic")

    file_path = tmp_path / "file.txt"
    file_path.write_text("data")

    # monkeypatch magic.from_file to return text/plain
    monkeypatch.setattr("magic.from_file", lambda *args, **kwargs: "text/plain")
    valid, _ = validate_file_type(str(file_path))
    assert valid is True

    monkeypatch.setattr("magic.from_file", lambda *args, **kwargs: "application/pdf")
    valid, msg = validate_file_type(str(file_path))
    assert valid is False
    assert "doesn't match" in msg
