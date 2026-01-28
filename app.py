"""
쇼특허 (Short-Cut) v3.0 - Streamlit Web Application with Streaming
====================================================================
AI 기반 특허 선행 기술 조사 시스템

Features:
- Zero-latency startup with @st.cache_resource
- Pre-loaded FAISS + BM25 hybrid index
- LLM Streaming response for real-time analysis
- Async pipeline with ThreadPoolExecutor

Team: 뀨💕
License: MIT
"""

import streamlit as st
import asyncio
import nest_asyncio
import sys
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Apply nest_asyncio for Streamlit compatibility
nest_asyncio.apply()

# Add src to path for imports
SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="쇼특허 (Short-Cut) v3.0",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS for Modern Design
# =============================================================================

st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Metric cards with dynamic colors */
    .metric-low {
        background: linear-gradient(135deg, #1a472a 0%, #2d5016 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #2d5016;
    }
    .metric-medium {
        background: linear-gradient(135deg, #5c4a1f 0%, #6b5b1f 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #6b5b1f;
    }
    .metric-high {
        background: linear-gradient(135deg, #5c1a1a 0%, #6b1f1f 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #6b1f1f;
    }
    
    /* Risk badge */
    .risk-badge {
        font-size: 0.9rem;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
    }
    .risk-high { background: #dc3545; color: white; }
    .risk-medium { background: #ffc107; color: black; }
    .risk-low { background: #28a745; color: white; }
    
    /* Analysis section */
    .analysis-section {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #4a90d9;
    }
    
    /* Streaming text animation */
    .streaming-text {
        border-left: 3px solid #4a90d9;
        padding-left: 1rem;
        animation: pulse 1s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { border-left-color: #4a90d9; }
        50% { border-left-color: #1a5490; }
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Cached Resource Loading (Zero-Latency Startup)
# =============================================================================

@st.cache_resource
def load_db_client():
    """Load Pinecone + BM25 hybrid client."""
    from vector_db import PineconeClient
    
    # PineconeClient automatically connects to serverless index
    # and loads local BM25 index if available
    client = PineconeClient()
    client.load_local()  # Load local BM25 index and metadata cache
    
    try:
        stats = client.get_stats()
    except:
        stats = {"total_vectors": 0, "initialized": False}
        
    return client, stats


@st.cache_resource
def get_openai_api_key():
    """Get OpenAI API key from environment."""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    return os.environ.get("OPENAI_API_KEY", "")


@st.cache_resource
def get_executor():
    """Get thread pool executor for async operations."""
    return ThreadPoolExecutor(max_workers=4)


# Load resources at startup
DB_CLIENT, DB_STATS = load_db_client()
OPENAI_API_KEY = get_openai_api_key()
EXECUTOR = get_executor()


# =============================================================================
# Session State Initialization
# =============================================================================

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "streaming_text" not in st.session_state:
    st.session_state.streaming_text = ""


# =============================================================================
# Helper Functions
# =============================================================================

def get_risk_color(risk_level: str) -> tuple:
    """Get color scheme based on risk level."""
    colors = {
        "high": ("#dc3545", "🔴", "metric-high"),
        "medium": ("#ffc107", "🟡", "metric-medium"),
        "low": ("#28a745", "🟢", "metric-low"),
    }
    return colors.get(risk_level.lower(), ("#6c757d", "⚪", "metric-low"))


def get_score_color(score: int) -> str:
    """Get color based on similarity score."""
    if score >= 70:
        return "#dc3545"
    elif score >= 40:
        return "#ffc107"
    else:
        return "#28a745"


def format_analysis_markdown(result: dict) -> str:
    """Format analysis result as downloadable markdown."""
    analysis = result.get("analysis", {})
    
    md = f"""# ⚡ 쇼특허 (Short-Cut) Analysis Report
> Generated: {result.get('timestamp', datetime.now().isoformat())}
> Search Type: {result.get('search_type', 'hybrid').upper()}

## 💡 User Idea
{result.get('user_idea', 'N/A')}

---

## 📊 Analysis Summary

### [1. 유사도 평가] Similarity Assessment
- **Score**: {analysis.get('similarity', {}).get('score', 0)}/100
- **Summary**: {analysis.get('similarity', {}).get('summary', 'N/A')}
- **Common Elements**: {', '.join(analysis.get('similarity', {}).get('common_elements', []))}
- **Evidence Patents**: {', '.join(analysis.get('similarity', {}).get('evidence', []))}

### [2. 침해 리스크] Infringement Risk
- **Risk Level**: {analysis.get('infringement', {}).get('risk_level', 'unknown').upper()}
- **Summary**: {analysis.get('infringement', {}).get('summary', 'N/A')}
- **Risk Factors**:
{chr(10).join(['  - ' + f for f in analysis.get('infringement', {}).get('risk_factors', [])])}
- **Evidence Patents**: {', '.join(analysis.get('infringement', {}).get('evidence', []))}

### [3. 회피 전략] Avoidance Strategy
- **Summary**: {analysis.get('avoidance', {}).get('summary', 'N/A')}
- **Strategies**:
{chr(10).join(['  - ' + s for s in analysis.get('avoidance', {}).get('strategies', [])])}
- **Alternatives**: {', '.join(analysis.get('avoidance', {}).get('alternatives', []))}

---

## 📌 Conclusion
{analysis.get('conclusion', 'N/A')}

---

## 📚 Referenced Patents
"""
    for patent in result.get("search_results", []):
        rrf = patent.get('rrf_score', 0)
        md += f"\n- **{patent.get('patent_id')}**: Score: {patent.get('grading_score', 0):.2f} | RRF: {rrf:.4f}"
    
    md += "\n\n---\n*Generated by 쇼특허 (Short-Cut) v3.0 | Team 뀨💕*"
    
    return md


def run_async_in_thread(coro):
    """Run async coroutine in a new event loop in thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def run_search_phase(agent, user_idea: str, use_hybrid: bool = True):
    """Run search and grading phase."""
    from patent_agent import PatentSearchResult
    
    # HyDE search
    hypothetical_claim, _ = await agent.hyde_search(user_idea, use_hybrid=use_hybrid)
    
    # Search with grading
    results = await agent.search_with_grading(user_idea, use_hybrid=use_hybrid)
    
    return hypothetical_claim, results


async def run_analysis_streaming(agent, user_idea: str, results, output_container):
    """Run streaming analysis and display in real-time."""
    full_text = ""
    placeholder = output_container.empty()
    
    async for token in agent.critical_analysis_stream(user_idea, results):
        full_text += token
        placeholder.markdown(full_text + "▌")  # Cursor effect
    
    placeholder.markdown(full_text)  # Final output without cursor
    return full_text


async def run_full_analysis(user_idea: str, status_container, streaming_container, use_hybrid: bool = True):
    """Run the complete patent analysis with streaming."""
    from patent_agent import PatentAgent, CriticalAnalysisResponse
    
    # Create agent with cached DB client
    agent = PatentAgent(db_client=DB_CLIENT)
    
    results = []
    analysis = None
    
    with status_container.status("🔍 특허 분석 중...", expanded=True) as status:
        # Step 1: HyDE
        status.write("📝 **Step 1/4**: HyDE - 가상 청구항 생성 중...")
        hypothetical_claim = await agent.generate_hypothetical_claim(user_idea)
        status.write(f"✅ 가상 청구항 생성 완료")
        status.write(f"```\n{hypothetical_claim[:200]}...\n```")
        
        # Step 2: Hybrid Search
        search_type = "Hybrid (Dense + BM25)" if use_hybrid else "Dense Only"
        status.write(f"🔎 **Step 2/4**: {search_type} 검색 중...")
        
        query_embedding = await agent.embed_text(hypothetical_claim)
        keywords = await agent.extract_keywords(user_idea + " " + hypothetical_claim)
        query_text = " ".join(keywords)
        
        if use_hybrid:
            search_results = await agent.db_client.async_hybrid_search(
                query_embedding, query_text, top_k=5
            )
        else:
            search_results = await agent.db_client.async_search(query_embedding, top_k=5)
        
        from patent_agent import PatentSearchResult
        results = []
        for r in search_results:
            results.append(PatentSearchResult(
                publication_number=r.patent_id,
                title=r.metadata.get("title", ""),
                abstract=r.metadata.get("abstract", r.content[:500]),
                claims=r.metadata.get("claims", ""),
                ipc_codes=[r.metadata.get("ipc_code", "")] if r.metadata.get("ipc_code") else [],
                similarity_score=r.score,
                dense_score=getattr(r, 'dense_score', 0.0),
                sparse_score=getattr(r, 'sparse_score', 0.0),
                rrf_score=getattr(r, 'rrf_score', 0.0),
            ))
        
        status.write(f"✅ {len(results)}개 유사 특허 발견")
        
        # Step 3: Grading
        status.write("📊 **Step 3/4**: 관련성 평가 중...")
        grading = await agent.grade_results(user_idea, results)
        status.write(f"✅ 평균 관련성 점수: {grading.average_score:.2f}")
        
        status.update(label="✅ 검색 완료! 분석 스트리밍 시작...", state="complete", expanded=False)
    
    # Step 4: Streaming Analysis
    streaming_container.markdown("### 🧠 실시간 분석 결과")
    streaming_container.caption("AI가 분석 내용을 실시간으로 생성합니다...")
    
    streamed_text = await run_analysis_streaming(agent, user_idea, results, streaming_container)
    
    # Also get structured analysis for result storage
    analysis = await agent.critical_analysis(user_idea, results)
    
    # Build result
    result = {
        "user_idea": user_idea,
        "search_results": [
            {
                "patent_id": r.publication_number,
                "title": r.title,
                "abstract": r.abstract,
                "claims": r.claims,
                "grading_score": r.grading_score,
                "grading_reason": r.grading_reason,
                "rrf_score": r.rrf_score,
            }
            for r in results
        ],
        "analysis": {
            "similarity": {
                "score": analysis.similarity.score,
                "common_elements": analysis.similarity.common_elements,
                "summary": analysis.similarity.summary,
                "evidence": analysis.similarity.evidence_patents,
            },
            "infringement": {
                "risk_level": analysis.infringement.risk_level,
                "risk_factors": analysis.infringement.risk_factors,
                "summary": analysis.infringement.summary,
                "evidence": analysis.infringement.evidence_patents,
            },
            "avoidance": {
                "strategies": analysis.avoidance.strategies,
                "alternatives": analysis.avoidance.alternative_technologies,
                "summary": analysis.avoidance.summary,
                "evidence": analysis.avoidance.evidence_patents,
            },
            "component_comparison": {
                "idea_components": analysis.component_comparison.idea_components,
                "matched_components": analysis.component_comparison.matched_components,
                "unmatched_components": analysis.component_comparison.unmatched_components,
                "risk_components": analysis.component_comparison.risk_components,
            },
            "conclusion": analysis.conclusion,
        },
        "streamed_analysis": streamed_text,
        "timestamp": datetime.now().isoformat(),
        "search_type": "hybrid" if use_hybrid else "dense",
    }
    
    return result


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("# ⚡ 쇼특허")
    st.markdown("### Short-Cut v3.0")
    st.divider()
    
    # System Status
    st.markdown("### ⚡ System Status")
    
    # API Status
    if OPENAI_API_KEY:
        st.success("✅ OpenAI API 연결됨")
    else:
        st.error("❌ OpenAI API 키 없음")
        st.info("`.env` 파일에 `OPENAI_API_KEY`를 설정하세요.")
    
    # DB Index Status
    if DB_CLIENT:
        st.success(f"✅ Hybrid 인덱스 로드됨")
        st.caption(f"   🌲 Pinecone: Connected")
        if DB_STATS.get('bm25_initialized'):
            st.caption(f"   📝 BM25 (Local): {DB_STATS.get('bm25_docs', 0):,}개 문서")
    else:
        st.warning("⚠️ DB 연결 실패")
        st.info("파이프라인을 실행하세요:\n`python src/pipeline.py --stage 5`")
    
    st.divider()
    
    # Search Options
    st.markdown("### 🔧 검색 옵션")
    use_hybrid = st.toggle("하이브리드 검색 (Dense + BM25)", value=True)
    if use_hybrid:
        st.caption("RRF 알고리즘으로 Dense와 Sparse 결과를 융합합니다.")
    else:
        st.caption("Dense (벡터) 검색만 사용합니다.")
    
    st.divider()
    
    # Analysis History
    st.markdown("### 📜 분석 히스토리")
    if st.session_state.analysis_history:
        for i, hist in enumerate(reversed(st.session_state.analysis_history[-5:])):
            with st.expander(f"#{len(st.session_state.analysis_history)-i}: {hist['user_idea'][:20]}..."):
                risk = hist.get('analysis', {}).get('infringement', {}).get('risk_level', 'unknown')
                score = hist.get('analysis', {}).get('similarity', {}).get('score', 0)
                search_type = hist.get('search_type', 'unknown')
                st.write(f"🎯 유사도: {score}/100")
                st.write(f"⚠️ 리스크: {risk.upper()}")
                st.write(f"🔍 검색: {search_type}")
                st.write(f"🕐 {hist.get('timestamp', 'N/A')[:10]}")
    else:
        st.caption("아직 분석 기록이 없습니다.")
    
    st.divider()
    
    # API Usage Guide
    st.markdown("### 💰 API 비용 가이드")
    st.caption("""
    **분석 1회 예상 비용**: ~$0.01-0.03
    
    - HyDE: gpt-4o-mini
    - Embed: text-embedding-3-small
    - Grading: gpt-4o-mini
    - Analysis: gpt-4o (Streaming)
    """)
    
    st.divider()
    st.markdown("##### Team 뀨💕")


# =============================================================================
# Main Content
# =============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>⚡ 쇼특허 (Short-Cut) v3.0</h1>
    <p style="font-size: 1.2rem; color: #888;">AI 기반 특허 선행 기술 조사 시스템</p>
    <p style="font-size: 0.9rem; color: #666;">Self-RAG | Hybrid Search | LLM Streaming</p>
</div>
""", unsafe_allow_html=True)

# Input Section
st.markdown("### 💡 아이디어 입력")
st.caption("특허로 출원하려는 아이디어를 설명해주세요. 유사 특허를 찾아 침해 리스크를 분석합니다.")

user_idea = st.text_area(
    label="아이디어 설명",
    placeholder="예: 딥러닝 기반 문서 요약 시스템으로, 긴 문서를 입력받아 핵심 내용을 추출하고 요약문을 생성합니다...",
    height=120,
    label_visibility="collapsed",
)

# Check if analysis is possible
can_analyze = (
    user_idea and 
    OPENAI_API_KEY and 
    DB_CLIENT
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyze_button = st.button(
        "🔍 특허 분석 시작",
        type="primary",
        use_container_width=True,
        disabled=not can_analyze,
    )

if not can_analyze and user_idea:
    if not OPENAI_API_KEY:
        st.warning("⚠️ OpenAI API 키를 설정하세요.")
    elif not DB_CLIENT:
        st.warning("⚠️ DB 클라이언트 초기화 실패.")

# Analysis Execution
if analyze_button and can_analyze:
    status_container = st.container()
    streaming_container = st.container()
    
    try:
        # Run async analysis using nest_asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            run_full_analysis(user_idea, status_container, streaming_container, use_hybrid=use_hybrid)
        )
        
        loop.close()
        
        # Store result
        st.session_state.current_result = result
        st.session_state.analysis_history.append(result)
        
    except Exception as e:
        st.error(f"❌ 분석 중 오류가 발생했습니다: {str(e)}")
        st.info("💡 OpenAI API 키를 확인하거나, 잠시 후 다시 시도해주세요.")


# =============================================================================
# Results Display
# =============================================================================

if st.session_state.current_result:
    result = st.session_state.current_result
    analysis = result.get("analysis", {})
    
    st.divider()
    st.markdown("## 📊 분석 결과")
    
    # Search Type Badge
    search_type = result.get("search_type", "hybrid")
    if search_type == "hybrid":
        st.success("🔀 하이브리드 검색 (Dense + BM25 + RRF)")
    else:
        st.info("🎯 Dense 검색")
    
    # Metric Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        score = analysis.get("similarity", {}).get("score", 0)
        score_color = get_score_color(score)
        st.metric(
            label="🎯 유사도 점수",
            value=f"{score}/100",
            delta="위험" if score >= 70 else ("주의" if score >= 40 else "양호"),
            delta_color="inverse" if score >= 40 else "normal",
        )
    
    with col2:
        risk_level = analysis.get("infringement", {}).get("risk_level", "unknown")
        color, emoji, css_class = get_risk_color(risk_level)
        st.metric(
            label="⚠️ 침해 리스크",
            value=f"{emoji} {risk_level.upper()}",
        )
    
    with col3:
        patent_count = len(result.get("search_results", []))
        st.metric(
            label="📚 참조 특허",
            value=f"{patent_count}건",
        )
    
    st.divider()
    
    # Analysis Report Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 종합 리포트", "🎯 유사도 분석", "⚠️ 침해 리스크", "🛡️ 회피 전략", "🔬 구성요소 대비"])
    
    with tab1:
        st.markdown("### 📌 결론")
        st.info(analysis.get("conclusion", "분석 결과가 없습니다."))
        
        # Download button
        md_content = format_analysis_markdown(result)
        st.download_button(
            label="📥 리포트 다운로드 (Markdown)",
            data=md_content,
            file_name=f"shortcut_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
        )
    
    with tab2:
        similarity = analysis.get("similarity", {})
        st.markdown(f"### 유사도 점수: {similarity.get('score', 0)}/100")
        st.markdown(f"**분석 요약**: {similarity.get('summary', 'N/A')}")
        
        st.markdown("**공통 기술 요소:**")
        for elem in similarity.get("common_elements", []):
            st.markdown(f"- {elem}")
        
        st.markdown("**근거 특허:**")
        for patent in similarity.get("evidence", []):
            st.code(patent)
    
    with tab3:
        infringement = analysis.get("infringement", {})
        risk = infringement.get("risk_level", "unknown")
        
        if risk == "high":
            st.error(f"🔴 **HIGH RISK** - 침해 가능성 높음")
        elif risk == "medium":
            st.warning(f"🟡 **MEDIUM RISK** - 주의 필요")
        else:
            st.success(f"🟢 **LOW RISK** - 침해 가능성 낮음")
        
        st.markdown(f"**분석 요약**: {infringement.get('summary', 'N/A')}")
        
        st.markdown("**위험 요소:**")
        for factor in infringement.get("risk_factors", []):
            st.markdown(f"- ⚠️ {factor}")
        
        st.markdown("**근거 특허:**")
        for patent in infringement.get("evidence", []):
            st.code(patent)
    
    with tab4:
        avoidance = analysis.get("avoidance", {})
        st.markdown(f"**권장 전략**: {avoidance.get('summary', 'N/A')}")
        
        st.markdown("**회피 설계 방안:**")
        for strategy in avoidance.get("strategies", []):
            st.markdown(f"- ✅ {strategy}")
        
        st.markdown("**대안 기술:**")
        for alt in avoidance.get("alternatives", []):
            st.markdown(f"- 💡 {alt}")
    
    with tab5:
        comp = analysis.get("component_comparison", {})
        st.markdown("### 🔬 구성요소 대비표")
        st.caption("사용자 아이디어의 구성요소와 선행 특허 비교 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 아이디어 구성요소")
            for c in comp.get("idea_components", []):
                st.markdown(f"- {c}")
        
        with col2:
            st.markdown("#### ✅ 일치 (선행 특허에 존재)")
            for c in comp.get("matched_components", []):
                st.markdown(f"- 🔴 {c}")
        
        st.divider()
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("#### 🆕 신규 (선행 특허에 없음)")
            for c in comp.get("unmatched_components", []):
                st.markdown(f"- 🟢 {c}")
            if not comp.get("unmatched_components"):
                st.caption("신규 구성요소가 없습니다.")
        
        with col4:
            st.markdown("#### ⚠️ 위험 구성요소")
            for c in comp.get("risk_components", []):
                st.markdown(f"- 🟡 {c}")
            if not comp.get("risk_components"):
                st.caption("특별히 위험한 구성요소가 없습니다.")
    
    # Referenced Patents
    st.divider()
    st.markdown("### 📚 참조된 선행 특허")
    
    for patent in result.get("search_results", []):
        rrf = patent.get('rrf_score', 0)
        with st.expander(f"📄 {patent.get('patent_id')} - Grade: {patent.get('grading_score', 0):.2f} | RRF: {rrf:.4f}"):
            st.markdown(f"**제목**: {patent.get('title', 'N/A')}")
            st.markdown(f"**관련성 평가**: {patent.get('grading_reason', 'N/A')}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**초록 (Abstract)**")
                st.caption(patent.get("abstract", "N/A")[:500] + "..." if len(patent.get("abstract", "")) > 500 else patent.get("abstract", "N/A"))
            with col2:
                st.markdown("**청구항 (Claims)**")
                st.caption(patent.get("claims", "N/A")[:500] + "..." if len(patent.get("claims", "")) > 500 else patent.get("claims", "N/A"))


# =============================================================================
# Footer
# =============================================================================

st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>⚡ 쇼특허 (Short-Cut) v3.0 | Self-RAG + Hybrid Search + Streaming</p>
    <p style="font-size: 0.8rem;">FAISS + BM25 + RRF | OpenAI API | Made with ❤️ by Team 뀨💕</p>
</div>
""", unsafe_allow_html=True)
