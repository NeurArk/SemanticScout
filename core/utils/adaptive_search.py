"""Adaptive search utilities for domain-agnostic similarity threshold calculation."""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Set, Tuple, Optional
import logging
import time

logger = logging.getLogger(__name__)


class AdaptiveSearchAnalyzer:
    """Analyzes queries and corpus to determine optimal similarity thresholds."""
    
    def __init__(self):
        self._corpus_stats: Optional[Dict] = None
        self._last_analysis_time: float = 0
        self._cache_duration = 300  # 5 minutes cache
        
    def get_adaptive_threshold(
        self, 
        query: str, 
        initial_results: int = 0,
        force_analysis: bool = False
    ) -> float:
        """
        Calculate adaptive threshold based on query complexity and corpus analysis.
        
        Args:
            query: The search query
            initial_results: Number of results from initial search (for auto-calibration)
            force_analysis: Force re-analysis of corpus
            
        Returns:
            Optimal similarity threshold between 0.15 and 0.45
        """
        # Get or compute corpus statistics
        corpus_stats = self._get_corpus_stats(force_analysis)
        
        # Analyze query complexity (0-1 score)
        complexity_score = self._analyze_query_complexity(query)
        
        # Analyze query specificity relative to corpus
        specificity_score = self._analyze_corpus_specificity(query, corpus_stats)
        
        # Base threshold varies by query length
        words = query.split()
        if len(words) <= 2:
            base_threshold = 0.20
        elif len(words) <= 4:
            base_threshold = 0.25
        else:
            base_threshold = 0.30
            
        # Adjust based on complexity and specificity
        threshold_adjustment = (complexity_score * 0.05) + (specificity_score * 0.10)
        threshold = base_threshold + threshold_adjustment
        
        # Auto-calibration if we have initial results
        if initial_results > 0 and corpus_stats.get('total_docs', 0) > 0:
            result_ratio = initial_results / corpus_stats['total_docs']
            if result_ratio < 0.01:  # Too few results
                threshold *= 0.8  # Lower threshold by 20%
            elif result_ratio > 0.3:  # Too many results
                threshold *= 1.2  # Increase threshold by 20%
                
        # Clamp to reasonable range
        final_threshold = max(0.15, min(0.45, threshold))
        
        logger.info(
            "Adaptive threshold for '%s': %.3f (complexity: %.2f, specificity: %.2f)",
            query, final_threshold, complexity_score, specificity_score
        )
        
        return final_threshold
    
    def _analyze_query_complexity(self, query: str) -> float:
        """
        Analyze query linguistic complexity (0-1 score).
        Higher scores indicate more complex/specific queries.
        """
        query_lower = query.lower()
        words = query_lower.split()
        
        complexity_factors = []
        
        # 1. Length factor (normalized to 8 words)
        length_factor = min(1.0, len(words) / 8)
        complexity_factors.append(length_factor)
        
        # 2. Question type analysis
        question_patterns = {
            # Simple queries (low complexity)
            r'^what is\b': 0.2,
            r'^who is\b': 0.3,
            r'^list\b': 0.2,
            r'^show\b': 0.2,
            
            # Intermediate queries
            r'^where\b': 0.4,
            r'^when\b': 0.4,
            r'^which\b': 0.5,
            
            # Complex queries (high complexity)
            r'^how\b': 0.6,
            r'^why\b': 0.7,
            r'^explain\b': 0.7,
            r'^compare\b': 0.8,
            r'^analyze\b': 0.9,
            r'^evaluate\b': 0.9,
        }
        
        question_factor = 0.3  # Default for non-question queries
        for pattern, score in question_patterns.items():
            if re.match(pattern, query_lower):
                question_factor = score
                break
        complexity_factors.append(question_factor)
        
        # 3. Specificity indicators
        specificity_words = {
            'specific', 'exact', 'particular', 'detailed', 'precise',
            'specifically', 'exactly', 'particularly', 'precisely'
        }
        has_specificity = any(word in specificity_words for word in words)
        complexity_factors.append(1.0 if has_specificity else 0.0)
        
        # 4. Presence of definite articles and demonstratives
        definite_indicators = {'the', 'this', 'that', 'these', 'those'}
        definite_count = sum(1 for word in words if word in definite_indicators)
        definite_factor = min(1.0, definite_count * 0.3)
        complexity_factors.append(definite_factor)
        
        # 5. Presence of technical/compound terms (multi-word or hyphenated)
        has_compound = any('-' in word or '_' in word for word in words)
        has_long_words = any(len(word) > 10 for word in words)
        technical_factor = 0.5 if (has_compound or has_long_words) else 0.0
        complexity_factors.append(technical_factor)
        
        # Average all factors
        return sum(complexity_factors) / len(complexity_factors)
    
    def _analyze_corpus_specificity(self, query: str, corpus_stats: Dict) -> float:
        """
        Analyze how specific the query is relative to the corpus vocabulary.
        Returns 0-1 score where higher means more specific to corpus.
        """
        if not corpus_stats or 'vocabulary' not in corpus_stats:
            return 0.0
            
        query_words = set(word.lower() for word in query.split() if len(word) > 2)
        
        # Check overlap with different vocabulary tiers
        common_overlap = len(query_words.intersection(corpus_stats['vocabulary']['common_terms']))
        domain_overlap = len(query_words.intersection(corpus_stats['vocabulary']['domain_terms']))
        rare_overlap = len(query_words.intersection(corpus_stats['vocabulary']['rare_terms']))
        
        # Weight different overlaps
        # Domain terms are most valuable, then rare, then common
        weighted_score = (
            (common_overlap * 0.1) +
            (domain_overlap * 0.5) +
            (rare_overlap * 0.3)
        )
        
        # Normalize by query length
        if len(query_words) > 0:
            specificity = min(1.0, weighted_score / len(query_words))
        else:
            specificity = 0.0
            
        return specificity
    
    def _get_corpus_stats(self, force_analysis: bool = False) -> Dict:
        """Get or compute corpus statistics."""
        current_time = time.time()
        
        # Check if we need to recompute
        if (not force_analysis and 
            self._corpus_stats is not None and 
            current_time - self._last_analysis_time < self._cache_duration):
            return self._corpus_stats
            
        # Import here to avoid circular dependency
        from core.vector_store import VectorStore
        vector_store = VectorStore()
        
        # Get all documents
        documents = vector_store.get_all_documents()
        
        if not documents:
            self._corpus_stats = {
                'total_docs': 0,
                'vocabulary': {
                    'common_terms': set(),
                    'domain_terms': set(),
                    'rare_terms': set()
                }
            }
        else:
            # Extract vocabulary from chunk metadata
            self._corpus_stats = self._extract_corpus_vocabulary(documents, vector_store)
            
        self._last_analysis_time = current_time
        return self._corpus_stats
    
    def _extract_corpus_vocabulary(
        self, 
        documents: List[Dict], 
        vector_store
    ) -> Dict:
        """Extract vocabulary statistics from corpus."""
        # Collect sample text from chunks
        all_text = []
        total_chunks = 0
        
        for doc in documents[:50]:  # Sample up to 50 documents
            doc_id = doc.get('document_id')
            if doc_id:
                # Get a few chunks from each document
                try:
                    chunks = vector_store.get_chunks_by_document_id(doc_id, limit=5)
                    for chunk in chunks:
                        all_text.append(chunk.content)
                        total_chunks += 1
                except Exception as e:
                    logger.warning(f"Failed to get chunks for document {doc_id}: {e}")
                    
        # Extract words
        words = []
        for text in all_text:
            # Simple word extraction (alphanumeric, length > 2)
            text_words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            words.extend(text_words)
            
        # Count frequencies
        word_freq = Counter(words)
        total_words = len(words)
        
        if total_words == 0:
            return {
                'total_docs': len(documents),
                'total_chunks': total_chunks,
                'vocabulary': {
                    'common_terms': set(),
                    'domain_terms': set(),
                    'rare_terms': set()
                }
            }
        
        # Categorize words by frequency
        # Common: top 20% most frequent
        # Domain: middle frequency (not too common, not too rare)
        # Rare: bottom 20%
        
        sorted_words = word_freq.most_common()
        top_20_percent = int(len(sorted_words) * 0.2)
        bottom_20_percent = int(len(sorted_words) * 0.8)
        
        common_terms = {word for word, _ in sorted_words[:top_20_percent]}
        rare_terms = {word for word, _ in sorted_words[bottom_20_percent:]}
        
        # Domain terms: frequency between 2 and 50, length > 4
        domain_terms = {
            word for word, count in word_freq.items()
            if 2 <= count <= 50 and len(word) > 4 and word not in common_terms
        }
        
        return {
            'total_docs': len(documents),
            'total_chunks': total_chunks,
            'vocabulary': {
                'common_terms': common_terms,
                'domain_terms': domain_terms,
                'rare_terms': rare_terms
            },
            'avg_chunk_length': total_words / max(total_chunks, 1)
        }
    
    def expand_query(self, query: str, corpus_stats: Optional[Dict] = None) -> str:
        """
        Expand short queries with contextual terms.
        
        Args:
            query: Original query
            corpus_stats: Optional corpus statistics
            
        Returns:
            Expanded query
        """
        words = query.lower().split()
        
        # Only expand very short queries
        if len(words) > 3:
            return query
            
        # Get corpus stats if not provided
        if corpus_stats is None:
            corpus_stats = self._get_corpus_stats()
            
        expanded_terms = []
        
        # Generic expansions for common short queries
        generic_expansions = {
            'attention': 'mechanism model transformer',
            'contract': 'agreement terms conditions',
            'report': 'analysis results findings',
            'summary': 'overview key points',
            'conclusion': 'results findings summary',
            'introduction': 'overview background context',
        }
        
        # Check for generic expansions
        for word in words:
            if word in generic_expansions:
                expanded_terms.extend(generic_expansions[word].split())
                
        # If we have corpus stats, add domain-specific terms
        if corpus_stats and 'vocabulary' in corpus_stats:
            domain_terms = corpus_stats['vocabulary'].get('domain_terms', set())
            # Find domain terms that might be related (simple heuristic)
            for word in words:
                for domain_term in list(domain_terms)[:20]:  # Check top domain terms
                    if word in domain_term or domain_term in word:
                        expanded_terms.append(domain_term)
                        
        # Combine original query with expansions
        if expanded_terms:
            expanded_query = f"{query} {' '.join(expanded_terms[:3])}"  # Limit expansion
            logger.info(f"Expanded query: '{query}' -> '{expanded_query}'")
            return expanded_query
            
        return query
    
    def reset_cache(self):
        """Reset the corpus analysis cache."""
        self._corpus_stats = None
        self._last_analysis_time = 0
        logger.info("Adaptive search cache reset")


# Global instance
adaptive_analyzer = AdaptiveSearchAnalyzer()