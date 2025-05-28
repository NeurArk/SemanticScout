"""Tests for adaptive search functionality."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch, MagicMock

from core.utils.adaptive_search import AdaptiveSearchAnalyzer
from core.models.document import DocumentChunk


class TestAdaptiveSearchAnalyzer:
    """Test cases for AdaptiveSearchAnalyzer."""
    
    def test_query_complexity_analysis(self):
        """Test query complexity scoring."""
        analyzer = AdaptiveSearchAnalyzer()
        
        # Simple queries should have low complexity (< 0.1)
        assert analyzer._analyze_query_complexity("what is") < 0.1
        assert analyzer._analyze_query_complexity("list documents") < 0.1
        
        # Medium complexity (0.15 - 0.35)
        assert 0.15 <= analyzer._analyze_query_complexity("where is the contract") < 0.35
        assert 0.15 <= analyzer._analyze_query_complexity("when was it signed") < 0.35
        assert 0.3 <= analyzer._analyze_query_complexity("compare the two agreements") < 0.4
        
        # High complexity (>= 0.45)
        assert analyzer._analyze_query_complexity("explain the specific details") >= 0.5
        assert analyzer._analyze_query_complexity("analyze the contract termination clauses") >= 0.45
    
    def test_adaptive_threshold_short_queries(self):
        """Test adaptive threshold for short queries."""
        analyzer = AdaptiveSearchAnalyzer()
        
        # Very short queries should get lower thresholds
        threshold = analyzer.get_adaptive_threshold("attention")
        assert 0.15 <= threshold <= 0.25
        
        threshold = analyzer.get_adaptive_threshold("contract")
        assert 0.15 <= threshold <= 0.25
    
    def test_adaptive_threshold_specific_queries(self):
        """Test adaptive threshold for specific queries."""
        analyzer = AdaptiveSearchAnalyzer()
        
        # Specific queries should get higher thresholds
        threshold = analyzer.get_adaptive_threshold("what are the specific termination clauses in the contract")
        assert 0.25 <= threshold <= 0.40
        
        threshold = analyzer.get_adaptive_threshold("explain the exact payment terms")
        assert 0.25 <= threshold <= 0.40
    
    def test_query_expansion(self):
        """Test query expansion functionality."""
        analyzer = AdaptiveSearchAnalyzer()
        
        # Short queries should be expanded
        expanded = analyzer.expand_query("attention")
        assert "attention" in expanded
        assert len(expanded.split()) > 1
        
        expanded = analyzer.expand_query("contract")
        assert "contract" in expanded
        assert any(word in expanded for word in ["agreement", "terms", "conditions"])
        
        # Long queries should not be expanded
        long_query = "what are the payment terms in the contract"
        assert analyzer.expand_query(long_query) == long_query
    
    @patch('core.vector_store.VectorStore')
    def test_corpus_analysis(self, mock_vector_store):
        """Test corpus vocabulary extraction."""
        analyzer = AdaptiveSearchAnalyzer()
        
        # Mock vector store and documents
        mock_store_instance = MagicMock()
        mock_vector_store.return_value = mock_store_instance
        
        mock_documents = [
            {"document_id": "doc1", "filename": "test.pdf"},
            {"document_id": "doc2", "filename": "contract.pdf"}
        ]
        mock_store_instance.get_all_documents.return_value = mock_documents
        
        # Mock chunks
        mock_chunks = [
            DocumentChunk(
                id="chunk1",
                document_id="doc1",
                content="This is a test document with some technical terms",
                chunk_index=0,
                start_char=0,
                end_char=50
            ),
            DocumentChunk(
                id="chunk2",
                document_id="doc1",
                content="Machine learning and artificial intelligence are important",
                chunk_index=1,
                start_char=50,
                end_char=100
            )
        ]
        mock_store_instance.get_chunks_by_document_id.return_value = mock_chunks
        
        # Force analysis
        stats = analyzer._get_corpus_stats(force_analysis=True)
        
        assert stats['total_docs'] == 2
        assert 'vocabulary' in stats
        assert isinstance(stats['vocabulary']['common_terms'], set)
        assert isinstance(stats['vocabulary']['domain_terms'], set)
    
    def test_corpus_specificity_analysis(self):
        """Test corpus specificity scoring."""
        analyzer = AdaptiveSearchAnalyzer()
        
        # Mock corpus stats
        corpus_stats = {
            'vocabulary': {
                'common_terms': {'document', 'test', 'data'},
                'domain_terms': {'contract', 'agreement', 'neural', 'transformer'},
                'rare_terms': {'proprietary', 'indemnification'}
            }
        }
        
        # Query with domain terms should have high specificity
        score = analyzer._analyze_corpus_specificity("contract agreement terms", corpus_stats)
        assert score > 0.3
        
        # Query with only common terms should have low specificity
        score = analyzer._analyze_corpus_specificity("test document data", corpus_stats)
        assert score < 0.3
        
        # Query with no corpus terms should have zero specificity
        score = analyzer._analyze_corpus_specificity("xyz abc qwe", corpus_stats)
        assert score == 0.0
    
    def test_cache_behavior(self):
        """Test cache behavior of corpus analysis."""
        analyzer = AdaptiveSearchAnalyzer()
        
        # Set cache with mock data
        analyzer._corpus_stats = {'test': 'data'}
        analyzer._last_analysis_time = float('inf')  # Far future
        
        # Should return cached data
        stats = analyzer._get_corpus_stats()
        assert stats == {'test': 'data'}
        
        # Reset cache
        analyzer.reset_cache()
        assert analyzer._corpus_stats is None
        assert analyzer._last_analysis_time == 0
    
    def test_threshold_ranges(self):
        """Test that thresholds stay within expected ranges."""
        analyzer = AdaptiveSearchAnalyzer()
        
        test_queries = [
            "a",  # Ultra short
            "what is",  # Short
            "where is the document",  # Medium
            "explain the specific technical details of the implementation",  # Long
            "analyze compare evaluate the comprehensive documentation",  # Complex
        ]
        
        for query in test_queries:
            threshold = analyzer.get_adaptive_threshold(query)
            assert 0.15 <= threshold <= 0.45, f"Threshold {threshold} out of range for query: {query}"
    
    def test_auto_calibration(self):
        """Test auto-calibration based on result count."""
        analyzer = AdaptiveSearchAnalyzer()
        
        # Mock corpus stats
        analyzer._corpus_stats = {
            'total_docs': 100, 
            'vocabulary': {
                'common_terms': set(),
                'domain_terms': set(),
                'rare_terms': set()
            }
        }
        analyzer._last_analysis_time = float('inf')
        
        # Normal threshold (no initial results)
        normal_threshold = analyzer.get_adaptive_threshold("test query")
        
        # Too few results but not zero - should lower threshold
        # Using 1 result out of 100 docs = 1% which triggers the < 0.01 condition
        few_results_threshold = analyzer.get_adaptive_threshold("test query", initial_results=1)
        # Since result_ratio = 0.01, this is not < 0.01, so it won't trigger
        # Let's use a larger corpus to trigger the condition
        analyzer._corpus_stats['total_docs'] = 1000
        few_results_threshold = analyzer.get_adaptive_threshold("test query", initial_results=5)
        # Now ratio = 5/1000 = 0.005 < 0.01, so it should lower by 20%
        assert few_results_threshold < normal_threshold  # Should be lowered due to low results
        
        # Too many results - should increase threshold  
        analyzer._corpus_stats['total_docs'] = 100  # Reset
        many_results_threshold = analyzer.get_adaptive_threshold("test query", initial_results=40)
        assert many_results_threshold > normal_threshold  # Should be increased due to many results