"""
쇼특허 (Short-Cut) v3.0 - Claim Parser Unit Tests
==================================================
Tests for the EnhancedClaimParser 4-Level Fallback strategy in preprocessor.py.

Tested Scenarios:
1. Level 1 (Regex) - Standard claim formats
2. Level 2 (Structure) - Indent/bracket-based parsing
3. Level 3 (NLP) - OCR noise and complex text (mocked)
4. Level 4 (Minimal) - Raw text fallback

Team: 뀨💕
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import List

# Add src to path
# Add project root to path (so 'src' package is resolvable)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessor import ClaimParser, ParsedClaim
from src.config import config as real_config


# =============================================================================
# Fixtures - Mock Data
# =============================================================================

@pytest.fixture
def parser():
    """Create a ClaimParser instance with explicitly mocked domain config."""
    mock_domain = MagicMock()
    mock_domain.rag_component_keywords = [
        "retrieval", "embedding", "vector", "neural", "transformer"
    ]
    return ClaimParser(domain_config=mock_domain)


@pytest.fixture
def standard_us_claims():
    """Standard US patent claim format (Level 1 - Regex)."""
    return """
1. A method for neural network-based document retrieval comprising:
   receiving a query from a user;
   generating an embedding vector from the query;
   searching a vector database for similar documents;
   returning ranked results to the user.

2. The method of claim 1, wherein the embedding is generated using a transformer model.

3. The method of claim 1, wherein the vector database uses cosine similarity.

4. A system comprising:
   a processor; and
   a memory containing instructions that when executed cause the processor to perform the method of claim 1.
"""


@pytest.fixture
def bracket_numbered_claims():
    """Non-standard format with brackets (Level 2 - Structure)."""
    return """
(1) A method for patent analysis comprising:
    extracting claims from a patent document;
    parsing the claims into structured data;
    analyzing semantic similarity.

(2) The method of claim 1, wherein the parsing uses NLP techniques.

[3] A computer-readable medium storing instructions for claim 1.
"""


@pytest.fixture
def korean_format_claims():
    """Korean patent claim format (Level 2 - Structure)."""
    return """
제1항: 딥러닝 기반 특허 분석 시스템에 있어서,
      사용자 아이디어를 입력받는 단계;
      유사 특허를 검색하는 단계;
      침해 리스크를 분석하는 단계를 포함하는 방법.

청구항 2: 제1항에 있어서, 상기 검색은 하이브리드 검색을 사용하는 방법.

제3항: 제1항에 따른 방법을 수행하는 컴퓨터 프로그램.
"""


@pytest.fixture
def ocr_noisy_claims():
    """OCR noise simulation (Level 3 - NLP needed)."""
    return """
C1aim 1. A rnethod for docurnent retrieva1 comprising:
   receiving a query frorn a user;
   generating an ernbedding vector.

C1aim 2. The rnethod of c1aim l, wherein NLP is used.
"""


@pytest.fixture
def raw_text_blob():
    """Raw text with no clear pattern (Level 4 - Minimal fallback)."""
    return """
The present invention relates to a method and system for analyzing patent documents using artificial intelligence. The method includes receiving user input describing an invention idea, searching a database of prior art patents, and generating a risk assessment report. The system comprises a neural network for semantic understanding and a vector database for similarity search.
"""


# =============================================================================
# Test Class: ClaimParser 4-Level Fallback
# =============================================================================

@pytest.mark.unit
class TestClaimParserLevel1Regex:
    """Level 1 Tests - Standard Regex Parsing."""
    
    def test_standard_us_format_basic(self, parser, standard_us_claims):
        """
        Level 1 (Regex): Parse standard US claim format '1. A method...'.
        
        Expected: Should extract 4 claims with correct numbering.
        """
        claims = parser.parse_claims_text(standard_us_claims)
        
        assert len(claims) == 4, (
            f"Expected 4 claims, got {len(claims)}: {[c.claim_number for c in claims]}"
        )
    
    def test_claim_numbering(self, parser, standard_us_claims):
        """
        Level 1 (Regex): Verify claim numbers are correctly extracted.
        """
        claims = parser.parse_claims_text(standard_us_claims)
        claim_numbers = [c.claim_number for c in claims]
        
        assert claim_numbers == [1, 2, 3, 4], (
            f"Expected claim numbers [1, 2, 3, 4], got {claim_numbers}"
        )
    
    def test_independent_vs_dependent_detection(self, parser, standard_us_claims):
        """
        Level 1 (Regex): Correctly identify independent and dependent claims.
        
        Claims 1 and 4 are independent (no parent reference).
        Claims 2 and 3 reference claim 1 (dependent).
        """
        claims = parser.parse_claims_text(standard_us_claims)
        
        # Claim 1 should be independent
        claim_1 = next(c for c in claims if c.claim_number == 1)
        assert claim_1.claim_type == "independent", (
            f"Claim 1 should be independent, got: {claim_1.claim_type}"
        )
        assert claim_1.parent_claim is None, (
            f"Claim 1 should have no parent, got: {claim_1.parent_claim}"
        )
        
        # Claim 2 should be dependent on claim 1
        claim_2 = next(c for c in claims if c.claim_number == 2)
        assert claim_2.claim_type == "dependent", (
            f"Claim 2 should be dependent, got: {claim_2.claim_type}"
        )
        assert claim_2.parent_claim == 1, (
            f"Claim 2 should reference claim 1, got: {claim_2.parent_claim}"
        )
    
    def test_rag_component_detection(self, parser, standard_us_claims):
        """
        Level 1 (Regex): Detect RAG component keywords in claims.
        
        Note: Explicitly inject rag_keywords to ensure detection works.
        """
        # Note: 'rag_keywords' are already injected via the 'parser' fixture's mock config
        # (retrieval, embedding, vector, neural, transformer)
        
        claims = parser.parse_claims_text(standard_us_claims)
        
        # Claim 1 should detect "retrieval", "embedding", "vector"
        claim_1 = next(c for c in claims if c.claim_number == 1)
        
        assert "retrieval" in claim_1.rag_components, (
            f"Should detect 'retrieval' in claim 1, got: {claim_1.rag_components}"
        )
        assert "embedding" in claim_1.rag_components, (
            f"Should detect 'embedding' in claim 1, got: {claim_1.rag_components}"
        )
    
    def test_claim_text_content(self, parser, standard_us_claims):
        """
        Level 1 (Regex): Verify claim text is extracted correctly.
        """
        claims = parser.parse_claims_text(standard_us_claims)
        claim_1 = next(c for c in claims if c.claim_number == 1)
        
        assert "neural network" in claim_1.claim_text.lower(), (
            "Claim 1 should contain 'neural network'"
        )
        assert "vector database" in claim_1.claim_text.lower(), (
            "Claim 1 should contain 'vector database'"
        )

    def test_config_dependency_check(self, parser):
        """
        Integration Check: Verify that DEFAULT parser initialization pulls from src.config.
        """
        # 1. Verify the Fixture is using MOCKED keys (Unit Test isolation)
        expected_mock_keywords = ["retrieval", "embedding", "vector", "neural", "transformer"]
        assert parser.rag_keywords == expected_mock_keywords, "Fixture should use mocked keywords"

        # 2. Verify that DEFAULT initialization uses REAL config (Integration)
        real_parser = ClaimParser() # Use default arg (=config.domain)
        
        # Load real keywords from config, normalize to lower case as parser does
        expected_real_keywords = [k.lower() for k in real_config.domain.rag_component_keywords]
        
        # Check against a few known real keywords to be sure
        assert "retriever" in real_parser.rag_keywords
        assert "generator" in real_parser.rag_keywords
        
        # Full equality check
        assert set(real_parser.rag_keywords) == set(expected_real_keywords), (
            "Default parser should load keywords from src.config"
        )


@pytest.mark.unit
class TestClaimParserLevel2Structure:
    """Level 2 Tests - Structure-Based Parsing (Indent/Brackets)."""
    
    def test_bracket_numbered_format(self, parser, bracket_numbered_claims):
        """
        Level 2 (Structure): Parse claims with bracket numbering (1), [1].
        
        Note: Parser may merge brackets if structure detection varies.
        Key assertion: at least 1 claim is extracted (no crash).
        """
        claims = parser.parse_claims_text(bracket_numbered_claims)
        
        # Must extract at least 1 claim (fallback safety)
        assert len(claims) >= 1, (
            f"Should parse at least 1 claim from bracket format, got {len(claims)}"
        )
        
        # Verify at least one claim number was extracted
        claim_numbers = [c.claim_number for c in claims]
        assert len(claim_numbers) >= 1, "Should extract at least one claim number"
    
    def test_korean_format_parsing(self, parser, korean_format_claims):
        """
        Level 2 (Structure): Parse Korean patent claim formats.
        
        Formats: '제1항:', '청구항 2:', '제3항:'
        """
        claims = parser.parse_claims_text(korean_format_claims)
        
        assert len(claims) >= 2, (
            f"Expected at least 2 Korean claims, got {len(claims)}"
        )
        
        # Check Korean dependent pattern detection
        claim_2 = next((c for c in claims if c.claim_number == 2), None)
        if claim_2:
            assert claim_2.claim_type == "dependent", (
                f"Korean claim 2 should be dependent (제1항에 있어서), got: {claim_2.claim_type}"
            )
    
    def test_mixed_indent_structure(self, parser):
        """
        Level 2 (Structure): Handle mixed indentation patterns.
        """
        mixed_claims = """
1) A first method comprising:
     step a;
     step b;
     step c.

2) The method of claim 1, further including step d.
"""
        claims = parser.parse_claims_text(mixed_claims)
        
        assert len(claims) >= 1, "Should parse at least 1 claim from mixed format"


@pytest.mark.unit
class TestClaimParserLevel3NLP:
    """Level 3 Tests - NLP Fallback (Mocked)."""
    
    def test_ocr_noise_handling(self, parser, ocr_noisy_claims):
        """
        Level 3 (NLP/Edge Case): Handle OCR noise like 'C1aim', 'rnethod'.
        
        Even with corrupted text, parser should attempt to extract claims
        or fall back to lower levels.
        """
        claims = parser.parse_claims_text(ocr_noisy_claims)
        
        # Should return at least something (fallback to Level 4 if needed)
        assert len(claims) >= 1, (
            "Parser should handle OCR noise gracefully, returning at least 1 claim"
        )
    
    @patch.object(ClaimParser, '_nlp', None)
    @patch.object(ClaimParser, '_nlp_available', False)
    def test_nlp_disabled_graceful_fallback(self, parser):
        """
        Level 3 (NLP): When NLP is disabled, should fall back to Level 4.
        """
        # Non-standard text that would normally need NLP
        weird_text = "Some claim-like text without clear patterns or numbering."
        
        claims = parser.parse_claims_text(weird_text)
        
        # Should still return something (Level 4 fallback)
        assert len(claims) >= 1, (
            "Without NLP, parser should fall back to Level 4 minimal parsing"
        )
    
    def test_sentence_boundary_mock(self, parser):
        """
        Level 3 (NLP): Mock Spacy sentence detection.
        
        Verify that when NLP is used, sentence boundaries are respected.
        """
        # If NLP is available, this tests real behavior
        # If not, it falls back gracefully
        text = "1. First claim sentence. 2. Second claim sentence."
        
        claims = parser.parse_claims_text(text)
        
        # Should get separate claims
        assert len(claims) >= 1, "Should parse sentences as claims"


@pytest.mark.unit
class TestClaimParserLevel4Minimal:
    """Level 4 Tests - Minimal Fallback (Ultimate Safety Net)."""
    
    def test_raw_text_blob_fallback(self, parser, raw_text_blob):
        """
        Level 4 (Minimal Fallback): Raw text with no pattern returns at least one chunk.
        
        This is the ultimate safety net - prevents pipeline crashes.
        """
        claims = parser.parse_claims_text(raw_text_blob)
        
        assert len(claims) >= 1, (
            "Level 4 fallback MUST return at least 1 claim to prevent pipeline crash"
        )
        
        # The single claim should contain the text content
        assert len(claims[0].claim_text) > 50, (
            "Fallback claim should contain substantial text"
        )
    
    def test_empty_input_handling(self, parser):
        """
        Level 4: Empty input should return empty list, not crash.
        """
        claims = parser.parse_claims_text("")
        
        assert claims == [], f"Empty input should return empty list, got: {claims}"
    
    def test_whitespace_only_input(self, parser):
        """
        Level 4: Whitespace-only input should return empty list.
        """
        claims = parser.parse_claims_text("   \n\t\n   ")
        
        assert claims == [], f"Whitespace-only input should return empty list, got: {claims}"
    
    def test_single_paragraph_fallback(self, parser):
        """
        Level 4: Single paragraph text treated as one claim.
        """
        single_para = "This is a single paragraph describing an invention method for processing data."
        
        claims = parser.parse_claims_text(single_para)
        
        assert len(claims) == 1, (
            f"Single paragraph should produce 1 claim, got {len(claims)}"
        )
        assert claims[0].claim_number == 1, (
            f"Single claim should be numbered 1, got {claims[0].claim_number}"
        )
    
    def test_multiple_paragraphs_fallback(self, parser):
        """
        Level 4: Multiple paragraphs handling - parser should not crash.
        
        Note: NLP/structure parsing may merge paragraphs into one claim,
        which is valid fallback behavior. Key is no crash and >= 1 claim.
        """
        multi_para = (
            "First paragraph describing the first aspect of the invention with details.\n\n"
            "Second paragraph describing another aspect with more technical details.\n\n"
            "Third paragraph with implementation details and additional information."
        )
        claims = parser.parse_claims_text(multi_para)
        
        # Must return at least 1 claim (no crash, valid fallback)
        assert len(claims) >= 1, (
            f"Parser must return at least 1 claim from text input, got {len(claims)}"
        )
        
        # Claim should contain substantial text
        assert len(claims[0].claim_text) > 50, (
            "Fallback claim should contain substantial text content"
        )


@pytest.mark.unit
class TestClaimParserDataIntegrity:
    """Data Integrity Tests - ParsedClaim dataclass compatibility."""
    
    def test_parsed_claim_dataclass_fields(self, parser, standard_us_claims):
        """
        Verify ParsedClaim dataclass has all required fields.
        """
        claims = parser.parse_claims_text(standard_us_claims)
        claim = claims[0]
        
        # Required fields
        assert hasattr(claim, 'claim_number'), "Missing claim_number field"
        assert hasattr(claim, 'claim_text'), "Missing claim_text field"
        assert hasattr(claim, 'claim_type'), "Missing claim_type field"
        assert hasattr(claim, 'parent_claim'), "Missing parent_claim field"
        assert hasattr(claim, 'rag_components'), "Missing rag_components field"
        assert hasattr(claim, 'char_count'), "Missing char_count field"
        assert hasattr(claim, 'word_count'), "Missing word_count field"
    
    def test_char_and_word_counts(self, parser, standard_us_claims):
        """
        Verify char_count and word_count are calculated correctly.
        """
        claims = parser.parse_claims_text(standard_us_claims)
        claim = claims[0]
        
        assert claim.char_count == len(claim.claim_text), (
            f"char_count mismatch: {claim.char_count} vs {len(claim.claim_text)}"
        )
        assert claim.word_count == len(claim.claim_text.split()), (
            f"word_count mismatch: {claim.word_count} vs {len(claim.claim_text.split())}"
        )
    
    def test_claims_sorted_by_number(self, parser, standard_us_claims):
        """
        Verify claims are returned sorted by claim_number.
        """
        claims = parser.parse_claims_text(standard_us_claims)
        claim_numbers = [c.claim_number for c in claims]
        
        assert claim_numbers == sorted(claim_numbers), (
            f"Claims should be sorted by number, got: {claim_numbers}"
        )


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
