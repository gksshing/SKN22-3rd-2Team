"""
쇼특허 (Short-Cut) v3.0 - Hybrid Search (RRF) Unit Tests
=========================================================
Tests for the RRF (Reciprocal Rank Fusion) algorithm in vector_db.py.

Tested Scenarios:
1. Cross-Rank Verification - Documents ranked high in one search appear in top results
2. Weighting Logic - Symmetric vs asymmetric weight behavior
3. Edge Cases - Empty inputs handling

Team: 뀨💕
"""

import pytest
import sys
import copy
from pathlib import Path
from unittest.mock import MagicMock, patch
from collections import defaultdict
from typing import List, Tuple, Dict, Any

# Add src to path
# Add project root to path (so 'src' package is resolvable)
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Fixtures - Mock Data
# =============================================================================

@pytest.fixture
def mock_search_result():
    """Factory fixture to create mock SearchResult objects."""
    def _create(chunk_id: str, patent_id: str, score: float, content: str = ""):
        mock = MagicMock()
        mock.chunk_id = chunk_id
        mock.patent_id = patent_id
        mock.score = score
        mock.content = content
        mock.dense_score = score
        mock.sparse_score = 0.0
        mock.rrf_score = 0.0
        mock.metadata = {
            "title": f"Patent {patent_id}",
            "abstract": content,
            "ipc_code": "G06N",
        }
        return mock
    return _create


@pytest.fixture
def dense_search_results(mock_search_result):
    """
    Mock dense search results with Doc A at #1.
    Doc C is NOT in this list (simulating it wasn't found by dense search).
    """
    return [
        mock_search_result("doc_a", "US-001", 0.95, "Document A - Top in Dense"),
        mock_search_result("doc_d", "US-004", 0.85, "Document D"),
        mock_search_result("doc_e", "US-005", 0.75, "Document E"),
        mock_search_result("doc_f", "US-006", 0.65, "Document F"),
        mock_search_result("doc_g", "US-007", 0.55, "Document G"),
        mock_search_result("doc_h", "US-008", 0.45, "Document H"),
        mock_search_result("doc_i", "US-009", 0.35, "Document I"),
        mock_search_result("doc_j", "US-010", 0.25, "Document J"),
        mock_search_result("doc_k", "US-011", 0.15, "Document K"),
        mock_search_result("doc_x", "US-020", 0.05, "Document X - filler at #10"),
    ]


@pytest.fixture
def sparse_search_results():
    """
    Mock sparse (BM25) search results with Doc B at #1.
    Doc C is NOT in this list (simulating it wasn't found by sparse search).
    """
    return [
        ("doc_b", 15.0, {"patent_id": "US-002", "content": "Document B - Top in Sparse"}),
        ("doc_l", 12.0, {"patent_id": "US-012", "content": "Document L"}),
        ("doc_m", 10.0, {"patent_id": "US-013", "content": "Document M"}),
        ("doc_n", 8.0, {"patent_id": "US-014", "content": "Document N"}),
        ("doc_o", 6.0, {"patent_id": "US-015", "content": "Document O"}),
        ("doc_p", 5.0, {"patent_id": "US-016", "content": "Document P"}),
        ("doc_q", 4.0, {"patent_id": "US-017", "content": "Document Q"}),
        ("doc_r", 3.0, {"patent_id": "US-018", "content": "Document R"}),
        ("doc_s", 2.0, {"patent_id": "US-019", "content": "Document S"}),
        ("doc_y", 1.0, {"patent_id": "US-021", "content": "Document Y - filler at #10"}),
    ]


# =============================================================================
# RRF Algorithm Implementation (Extracted for Testing)
# =============================================================================

# =============================================================================
# Helper Functions - Import from source
# =============================================================================

from src.vector_db import compute_rrf  # Ensure this is imported properly



# =============================================================================
# Test Class: Hybrid Search RRF
# =============================================================================

@pytest.mark.unit
class TestHybridSearchRRF:
    """Test suite for RRF (Reciprocal Rank Fusion) algorithm."""
    
    def test_cross_rank_verification_top_tier(
        self,
        dense_search_results,
        sparse_search_results
    ):
        """
        Cross-Rank Verification (The 'Top-tier' Test):
        
        Setup:
        - Doc A: Ranked #1 in Dense Search, unranked in Sparse Search
        - Doc B: Ranked #1 in Sparse Search, unranked in Dense Search
        
        Expectation:
        After RRF fusion, both Doc A and Doc B must appear in the Top 3 results
        due to their high individual rankings (rank #1 in their respective search).
        """
        # Execute RRF fusion
        results = compute_rrf(
            dense_results=dense_search_results,
            sparse_results=sparse_search_results,
            dense_weight=0.5,
            sparse_weight=0.5,
            rrf_k=60,
            top_k=50,
        )
        
        # Get top 3 chunk IDs (results are SearchResult objects)
        top_3_ids = [r.chunk_id for r in results[:3]]
        
        # Assertions - Both #1 ranked docs should be in top 3
        assert "doc_a" in top_3_ids, (
            f"Doc A (Dense #1) should be in Top 3, but Top 3 is: {top_3_ids}"
        )
        assert "doc_b" in top_3_ids, (
            f"Doc B (Sparse #1) should be in Top 3, but Top 3 is: {top_3_ids}"
        )
    
    def test_symmetric_weighting(
        self,
        dense_search_results,
        sparse_search_results
    ):
        """
        Weighting Logic - Symmetric (0.5:0.5):
        
        With equal weights, documents ranked #1 in either search
        should receive equal contribution to their RRF score.
        """
        results = compute_rrf(
            dense_results=dense_search_results,
            sparse_results=sparse_search_results,
            dense_weight=0.5,
            sparse_weight=0.5,
            rrf_k=60,
            top_k=50,
        )
        
        # Find scores for doc_a and doc_b
        scores = {r.chunk_id: r.score for r in results}
        
        # With rrf_k=60, rank 0 contribution = weight / (60 + 0 + 1) = weight / 61
        # expected_contribution = 0.5 / 61
        
        # Doc A only has dense, Doc B only has sparse
        # Their single contributions should be nearly equal
        assert abs(scores["doc_a"] - scores["doc_b"]) < 0.001, (
            f"Symmetric weights: Doc A ({scores['doc_a']:.6f}) and Doc B ({scores['doc_b']:.6f}) "
            f"should have similar scores, difference: {abs(scores['doc_a'] - scores['doc_b']):.6f}"
        )
    
    def test_asymmetric_weighting_dense_heavy(
        self,
        dense_search_results,
        sparse_search_results
    ):
        """
        Weighting Logic - Asymmetric (0.8:0.2):
        
        With dense_weight=0.8, documents ranked high in Dense search
        should have higher RRF scores than those ranked high only in Sparse.
        """
        results = compute_rrf(
            dense_results=dense_search_results,
            sparse_results=sparse_search_results,
            dense_weight=0.8,
            sparse_weight=0.2,
            rrf_k=60,
            top_k=50, 
        )
        
        scores = {r.chunk_id: r.score for r in results}
        
        # Doc A (Dense #1) should score higher than Doc B (Sparse #1)
        assert scores["doc_a"] > scores["doc_b"], (
            f"With dense_weight=0.8, Doc A (Dense #1: {scores['doc_a']:.6f}) should score higher "
            f"than Doc B (Sparse #1: {scores['doc_b']:.6f})"
        )
        
        # Verify doc_a is #1 overall
        top_id = results[0].chunk_id
        assert top_id == "doc_a", (
            f"With dense_weight=0.8, Doc A should be #1, but #{1} is: {top_id}"
        )
    
    def test_asymmetric_weighting_sparse_heavy(
        self,
        dense_search_results,
        sparse_search_results
    ):
        """
        Weighting Logic - Asymmetric (0.2:0.8):
        
        With sparse_weight=0.8, documents ranked high in Sparse search
        should have higher RRF scores than those ranked high only in Dense.
        """
        results = compute_rrf(
            dense_results=dense_search_results,
            sparse_results=sparse_search_results,
            dense_weight=0.2,
            sparse_weight=0.8,
            rrf_k=60,
            top_k=50,
        )
        
        scores = {r.chunk_id: r.score for r in results}
        
        # Doc B (Sparse #1) should score higher than Doc A (Dense #1)
        assert scores["doc_b"] > scores["doc_a"], (
            f"With sparse_weight=0.8, Doc B (Sparse #1: {scores['doc_b']:.6f}) should score higher "
            f"than Doc A (Dense #1: {scores['doc_a']:.6f})"
        )
    
    def test_edge_case_empty_dense_results(self, sparse_search_results):
        """
        Edge Case - Empty Dense Results:
        
        When dense search returns no results, RRF should still work
        using only sparse results without crashing.
        """
        results = compute_rrf(
            dense_results=[],  # Empty
            sparse_results=sparse_search_results,
            dense_weight=0.5,
            sparse_weight=0.5,
        )
        
        # Should have results from sparse search
        assert len(results) > 0, "Should still have results from sparse search"
        
        # Top result should be doc_b (Sparse #1)
        assert results[0].chunk_id == "doc_b", (
            f"With no dense results, Sparse #1 (doc_b) should be top, got: {results[0].chunk_id}"
        )
    
    def test_edge_case_empty_sparse_results(self, dense_search_results):
        """
        Edge Case - Empty Sparse Results:
        
        When sparse search returns no results, RRF should still work
        using only dense results without crashing.
        """
        results = compute_rrf(
            dense_results=dense_search_results,
            sparse_results=[],  # Empty
            dense_weight=0.5,
            sparse_weight=0.5,
        )
        
        # Should have results from dense search
        assert len(results) > 0, "Should still have results from dense search"
        
        # Top result should be doc_a (Dense #1)
        assert results[0].chunk_id == "doc_a", (
            f"With no sparse results, Dense #1 (doc_a) should be top, got: {results[0].chunk_id}"
        )
    
    def test_edge_case_both_empty(self):
        """
        Edge Case - Both Search Results Empty:
        
        When both searches return no results, RRF should return
        an empty list without crashing.
        """
        results = compute_rrf(
            dense_results=[],
            sparse_results=[],
            dense_weight=0.5,
            sparse_weight=0.5,
        )
        
        assert len(results) == 0, "Empty inputs should return empty results"
        assert isinstance(results, list), "Should return a list type"
    
    def test_rrf_k_constant_effect(
        self,
        dense_search_results,
        sparse_search_results
    ):
        """
        RRF K Constant Effect:
        
        Higher k values reduce the impact of top-ranked documents.
        With k=60 (default), rank 0 contribution = 1/61
        With k=10 (lower), rank 0 contribution = 1/11 (much higher)
        """
        results_k60 = compute_rrf(
            dense_results=copy.deepcopy(dense_search_results),
            sparse_results=copy.deepcopy(sparse_search_results),
            rrf_k=60,
            top_k=50,
        )
        
        results_k10 = compute_rrf(
            dense_results=copy.deepcopy(dense_search_results),
            sparse_results=copy.deepcopy(sparse_search_results),
            rrf_k=10,
            top_k=50,
        )
        
        scores_k60 = {r.chunk_id: r.score for r in results_k60}
        scores_k10 = {r.chunk_id: r.score for r in results_k10}
        
        # With lower k, top-ranked documents get relatively higher scores
        # The gap between #1 and #10 should be larger with k=10
        # Using doc_x (rank #10 in dense) for comparison
        gap_k60 = scores_k60["doc_a"] - scores_k60.get("doc_x", 0)
        gap_k10 = scores_k10["doc_a"] - scores_k10.get("doc_x", 0)
        
        assert gap_k10 > gap_k60, (
            f"Lower k should create larger gaps. k=10 gap: {gap_k10:.6f}, k=60 gap: {gap_k60:.6f}"
        )


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
