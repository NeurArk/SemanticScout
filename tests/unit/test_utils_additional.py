"""Additional tests to improve utils coverage."""

import pytest
import time
from unittest.mock import Mock, patch

from core.utils.performance import measure_time
from core.utils.text_processing import clean_text, extract_sentences, calculate_overlap


class TestPerformanceUtils:
    """Test performance monitoring utilities."""

    def test_measure_time_decorator(self):
        """Test measure_time decorator."""
        
        @measure_time
        def slow_function():
            time.sleep(0.01)
            return "result"
        
        result, elapsed = slow_function()
        assert result == "result"
        assert elapsed >= 0.01
        assert isinstance(elapsed, float)

    def test_measure_time_with_args(self):
        """Test measure_time with function arguments."""
        
        @measure_time
        def add_numbers(a, b):
            return a + b
        
        result, elapsed = add_numbers(5, 3)
        assert result == 8
        assert elapsed >= 0
        assert isinstance(elapsed, float)

    def test_measure_time_with_exception(self):
        """Test measure_time when function raises exception."""
        
        @measure_time
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function()


class TestTextProcessingUtils:
    """Test text processing utilities."""

    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        dirty_text = "Hello   World!   How  are   you?"
        clean = clean_text(dirty_text)
        assert clean == "Hello World! How are you?"

    def test_clean_text_special_chars(self):
        """Test cleaning special characters."""
        text = "Hello@#$%World&*()2024"
        clean = clean_text(text)
        assert "@" not in clean
        assert "#" not in clean
        assert "&" not in clean

    def test_clean_text_newlines(self):
        """Test cleaning newlines and tabs."""
        text = "Hello\n\n\tWorld\r\nTest"
        clean = clean_text(text)
        assert "\n" not in clean
        assert "\t" not in clean
        assert "\r" not in clean

    def test_extract_sentences_basic(self):
        """Test basic sentence extraction."""
        text = "This is first sentence. This is second! And third?"
        sentences = extract_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "This is first sentence."
        assert sentences[1] == "This is second!"
        assert sentences[2] == "And third?"

    def test_extract_sentences_multiple_punctuation(self):
        """Test extraction with multiple punctuation marks."""
        text = "Really?! Yes. Of course..."
        sentences = extract_sentences(text)
        assert len(sentences) == 3

    def test_extract_sentences_empty(self):
        """Test extraction from empty text."""
        sentences = extract_sentences("")
        assert sentences == []

    def test_extract_sentences_no_punctuation(self):
        """Test extraction from text without punctuation."""
        text = "This text has no punctuation"
        sentences = extract_sentences(text)
        assert len(sentences) == 1
        assert sentences[0] == text

    def test_calculate_overlap_basic(self):
        """Test basic overlap calculation."""
        text = "This is a test text for overlap calculation"
        start, end = calculate_overlap(text, 10, 20, 5)
        assert start == 5  # 10 - 5
        assert end == 25   # 20 + 5

    def test_calculate_overlap_at_boundaries(self):
        """Test overlap at text boundaries."""
        text = "Short text"
        # Test at start
        start, end = calculate_overlap(text, 0, 5, 3)
        assert start == 0  # Can't go below 0
        assert end == 8
        
        # Test at end
        start, end = calculate_overlap(text, 5, 10, 3)
        assert start == 2
        assert end == 10  # Can't exceed text length

    def test_calculate_overlap_zero_overlap(self):
        """Test with zero overlap."""
        text = "Test text"
        start, end = calculate_overlap(text, 2, 5, 0)
        assert start == 2
        assert end == 5

    def test_calculate_overlap_large_overlap(self):
        """Test with overlap larger than chunk."""
        text = "A" * 100
        start, end = calculate_overlap(text, 40, 50, 20)
        assert start == 20
        assert end == 70