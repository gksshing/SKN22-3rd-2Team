"""
쇼특허 (Short-Cut) v3.0 - DeepEval RAG Quality Tests
=====================================================
RAG 파이프라인 품질 검증 테스트 (Faithfulness, Answer Relevancy)

Metrics:
1. FaithfulnessMetric - 답변이 검색된 컨텍스트에 근거하는지 검증
2. AnswerRelevancyMetric - 답변이 사용자 질문과 관련 있는지 검증

Team: 뀨💕
"""

import pytest
import asyncio
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
# Add project root to path (so 'src' package is resolvable)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load Env
from dotenv import load_dotenv
load_dotenv()

# Check for required environment variables
if not os.environ.get("OPENAI_API_KEY"):
    pytest.skip(
        "OPENAI_API_KEY not set. Skipping DeepEval tests.",
        allow_module_level=True
    )

# DeepEval imports
try:
    from deepeval import assert_test
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
except ImportError:
    pytest.skip(
        "deepeval not installed. Run: pip install deepeval",
        allow_module_level=True
    )

# Import PatentAgent
from src.patent_agent import PatentAgent


# =============================================================================
# Golden Dataset - AI/NLP Domain Test Cases
# =============================================================================
# 테스트 케이스는 우리 데이터의 도메인 키워드에 맞게 설계됨:
# - retrieval augmented generation
# - large language model
# - neural information retrieval
# - semantic search
# - document embedding
# - transformer attention
# - knowledge graph reasoning
# - prompt engineering

GOLDEN_DATASET: List[Dict[str, Any]] = [
    {
        "id": "test_001",
        "name": "RAG 기반 문서 검색 시스템",
        "query": """
        Please generate a comprehensive patent analysis report for the following idea
        (including prior art search, infringement risk, and avoidance strategy):
        A document search system utilizing Retrieval Augmented Generation technology.
        It converts user queries into vector embeddings, retrieves similar documents 
        from a vector database, and provides them as context to an LLM.
        It uses hybrid search (Dense + Sparse) and RRF fusion.
        """,
        "expected_topics": ["retrieval", "embedding", "vector", "search"],
    },
    {
        "id": "test_002",
        "name": "Semantic Search 엔진",
        "query": """
        Please generate a comprehensive patent analysis report for the following idea
        (including prior art search, infringement risk, and avoidance strategy):
        A semantic search engine based on Neural information retrieval.
        It embeds documents and queries using Transformer models and 
        retrieves semantically similar documents via cosine similarity.
        It provides more accurate results than traditional keyword search.
        """,
        "expected_topics": ["semantic", "transformer", "embedding", "neural"],
    },
    {
        "id": "test_003",
        "name": "LLM Fine-tuning 시스템",
        "query": """
        Please generate a comprehensive patent analysis report for the following idea
        (including prior art search, infringement risk, and avoidance strategy):
        A system for fine-tuning Large Language Models on specific domains.
        It applies quantization techniques for efficient inference and 
        generates optimized results via prompt engineering.
        """,
        "expected_topics": ["language model", "fine-tuning", "inference", "prompt"],
    },
]


# =============================================================================
# Test Configuration
# =============================================================================

# Metric thresholds
FAITHFULNESS_THRESHOLD = 0.7
RELEVANCY_THRESHOLD = 0.7

# Evaluation model
EVAL_MODEL = "gpt-4o-mini"  # Cost-effective for testing


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def patent_agent():
    """Initialize PatentAgent (expensive, so module-scoped)."""
    try:
        agent = PatentAgent()
        return agent
    except Exception as e:
        pytest.skip(f"Failed to initialize PatentAgent: {e}")


# =============================================================================
# Helper Functions
# =============================================================================

def extract_retrieval_context(search_results: List[Dict]) -> List[str]:
    """
    Extract retrieval context from search results.
    
    Combines abstract and claims from each patent for full context.
    This is critical for DeepEval FaithfulnessMetric accuracy.
    """
    context = []
    for result in search_results:
        patent_id = result.get("patent_id", "Unknown")
        title = result.get("title", "")
        abstract = result.get("abstract", "")
        claims = result.get("claims", "")
        
        # Build comprehensive context string
        context_parts = [f"Patent {patent_id}: {title}"]
        
        if abstract:
            # Truncate abstract if too long (DeepEval has token limits, but 4o-mini handles 128k)
            # Increased from 1500 to 4000 chars to cover full context
            abstract_truncated = abstract[:4000] if len(abstract) > 4000 else abstract
            context_parts.append(f"Abstract: {abstract_truncated}")
        
        if claims:
            # Truncate claims if too long
            # Increased from 2000 to 5000 chars to ensure critical claims are included
            claims_truncated = claims[:5000] if len(claims) > 5000 else claims
            context_parts.append(f"Claims: {claims_truncated}")
        
        if result.get("grading_reason"):
            context_parts.append(f"Relevance: {result['grading_reason']}")
        
        context.append(" | ".join(context_parts))
    
    return context


def extract_actual_output(analysis: Dict) -> str:
    """
    Extract actual output string from analysis result.
    
    Combines key analysis sections into a single string.
    """
    parts = []
    
    # Similarity
    sim = analysis.get("similarity", {})
    if sim.get("summary"):
        parts.append(f"유사도 분석: {sim['summary']} (점수: {sim.get('score', 'N/A')})")
    
    # Infringement
    inf = analysis.get("infringement", {})
    if inf.get("summary"):
        parts.append(f"침해 리스크: {inf['summary']} (레벨: {inf.get('risk_level', 'N/A')})")
    
    # Avoidance
    avoid = analysis.get("avoidance", {})
    if avoid.get("summary"):
        parts.append(f"회피 전략: {avoid['summary']}")
    
    # Conclusion
    if analysis.get("conclusion"):
        parts.append(f"결론: {analysis['conclusion']}")
    
    return " | ".join(parts) if parts else "No analysis available"


async def run_agent_analysis(agent: PatentAgent, query: str) -> Dict[str, Any]:
    """Run PatentAgent analysis asynchronously."""
    return await agent.analyze(query, use_hybrid=True, stream=False)


# =============================================================================
# DeepEval Metric Instances
# =============================================================================

@pytest.fixture(scope="module")
def faithfulness_metric():
    """Create FaithfulnessMetric instance."""
    return FaithfulnessMetric(
        threshold=FAITHFULNESS_THRESHOLD,
        model=EVAL_MODEL,
        include_reason=True,
    )


@pytest.fixture(scope="module")
def relevancy_metric():
    """Create AnswerRelevancyMetric instance."""
    return AnswerRelevancyMetric(
        threshold=RELEVANCY_THRESHOLD,
        model=EVAL_MODEL,
        include_reason=True,
    )


# =============================================================================
# Test Class: RAG Quality Evaluation
# =============================================================================

class TestRAGQuality:
    """
    RAG 파이프라인 품질 테스트 클래스.
    
    DeepEval의 FaithfulnessMetric과 AnswerRelevancyMetric을 사용하여
    PatentAgent의 분석 결과가 검색된 컨텍스트에 충실한지 검증합니다.
    """
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.parametrize("test_case", GOLDEN_DATASET, ids=lambda tc: tc["id"])
    async def test_rag_quality(
        self,
        test_case: Dict[str, Any],
        patent_agent: PatentAgent,
        faithfulness_metric: FaithfulnessMetric,
        relevancy_metric: AnswerRelevancyMetric,
        record_property,
    ):
        """
        RAG 품질 테스트: Faithfulness + Answer Relevancy.
        
        Args:
            test_case: Golden dataset의 테스트 케이스
            patent_agent: PatentAgent 인스턴스
            faithfulness_metric: 충실도 메트릭
            relevancy_metric: 관련성 메트릭
            record_property: pytest fixture for custom report attributes
        """
        print(f"\n{'='*60}")
        print(f"🧪 Test Case: {test_case['name']}")
        print(f"{'='*60}")
        
        query = test_case["query"].strip()
        
        # Step 1: Run PatentAgent analysis
        try:
            result = await run_agent_analysis(patent_agent, query)
        except Exception as e:
            pytest.fail(f"PatentAgent.analyze() failed: {e}")
        
        # Check for errors
        if "error" in result:
            pytest.skip(f"No patents found: {result['error']}")
        
        # Step 2: Extract components for LLMTestCase
        search_results = result.get("search_results", [])
        analysis = result.get("analysis", {})
        
        # Input: User's idea
        input_text = query
        
        # Actual Output: Analysis conclusion/summary
        actual_output = extract_actual_output(analysis)
        
        # Retrieval Context: Patent abstracts/claims
        retrieval_context = extract_retrieval_context(search_results)
        
        print(f"\n📥 Input Query: {input_text[:100]}...")
        print(f"📤 Actual Output: {actual_output[:200]}...")
        print(f"📚 Context Count: {len(retrieval_context)}")
        
        # Step 3: Create LLMTestCase
        llm_test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            retrieval_context=retrieval_context,
        )
        
        # Step 4: Measure and assert with DeepEval metrics
        print("\n🔍 Running DeepEval metrics...")
        
        # Measure Faithfulness (this computes the score)
        faithfulness_metric.measure(llm_test_case)
        faith_score = faithfulness_metric.score
        faith_reason = faithfulness_metric.reason
        
        # Record to XML report
        record_property("faithfulness_score", faith_score)
        record_property("faithfulness_reason", faith_reason or "N/A")
        
        print(f"   📊 Faithfulness Score: {faith_score:.2f} (threshold: {FAITHFULNESS_THRESHOLD})")
        if faith_reason:
            print(f"      Reason: {faith_reason[:150]}...")
        
        # Assert Faithfulness
        if faith_score < FAITHFULNESS_THRESHOLD:
            raise AssertionError(f"Faithfulness score {faith_score:.2f} < {FAITHFULNESS_THRESHOLD}")
        print(f"   ✅ Faithfulness: PASSED")
        
        # Measure Answer Relevancy
        relevancy_metric.measure(llm_test_case)
        rel_score = relevancy_metric.score
        rel_reason = relevancy_metric.reason
        
        # Record to XML report
        record_property("relevancy_score", rel_score)
        record_property("relevancy_reason", rel_reason or "N/A")

        print(f"   📊 Answer Relevancy Score: {rel_score:.2f} (threshold: {RELEVANCY_THRESHOLD})")
        if rel_reason:
            print(f"      Reason: {rel_reason[:150]}...")
        
        # Assert Relevancy
        if rel_score < RELEVANCY_THRESHOLD:
            raise AssertionError(f"Answer Relevancy score {rel_score:.2f} < {RELEVANCY_THRESHOLD}")
        print(f"   ✅ Answer Relevancy: PASSED")
        
        print(f"\n{'='*60}")
        print(f"✅ Test Case '{test_case['name']}' PASSED")
        print(f"{'='*60}")
    



# =============================================================================
# Standalone Test Functions (Alternative to Class)
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.integration
async def test_single_query_quality(patent_agent: PatentAgent):
    """
    단일 쿼리에 대한 빠른 품질 검증 테스트.
    
    CI/CD 파이프라인에서 빠르게 실행할 수 있는 경량 테스트.
    """
    query = "Natural Language Processing based Patent Search System"
    
    result = await patent_agent.analyze(query, use_hybrid=True)
    
    # Basic assertions (not DeepEval)
    assert "analysis" in result, "Analysis should be present in result"
    assert "search_results" in result, "Search results should be present"
    assert result.get("analysis", {}).get("conclusion"), "Conclusion should not be empty"
    
    print(f"✅ Single query test passed")
    print(f"   - Found {len(result.get('search_results', []))} patents")
    print(f"   - Similarity score: {result['analysis']['similarity'].get('score')}")


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
        "--asyncio-mode=auto",
    ])
