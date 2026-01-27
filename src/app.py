"""
Patent Guard v2.0 - Streamlit Web Application
==============================================
Professional patent analysis demo using Self-RAG with HyDE, Grading, and CoT Analysis.

Author: Patent Guard Team
License: MIT
"""

import streamlit as st
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from patent_agent import PatentAgent, OPENAI_API_KEY

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Patent Guard v2.0",
    page_icon="🛡️",
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
    
    /* Patent card */
    .patent-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
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
# Session State Initialization
# =============================================================================

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None


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
        return "#dc3545"  # Red - High risk
    elif score >= 40:
        return "#ffc107"  # Yellow - Medium
    else:
        return "#28a745"  # Green - Low


def format_analysis_markdown(result: dict) -> str:
    """Format analysis result as downloadable markdown."""
    analysis = result.get("analysis", {})
    
    md = f"""# 🛡️ Patent Guard Analysis Report
> Generated: {result.get('timestamp', datetime.now().isoformat())}

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
        md += f"\n- **{patent.get('patent_id')}**: {patent.get('title', 'N/A')} (Score: {patent.get('grading_score', 0):.2f})"
    
    md += "\n\n---\n*Generated by Patent Guard v2.0 - 뀨💕*"
    
    return md


async def run_analysis(user_idea: str, status_container) -> dict:
    """Run the patent analysis with status updates."""
    agent = PatentAgent()
    
    with status_container.status("🔍 특허 분석 중...", expanded=True) as status:
        # Step 1: HyDE
        status.write("📝 **Step 1/3**: HyDE - 가상 청구항 생성 중...")
        hypothetical_claim = await agent.generate_hypothetical_claim(user_idea)
        status.write(f"✅ 가상 청구항 생성 완료")
        status.write(f"```\n{hypothetical_claim[:200]}...\n```")
        
        # Step 2: BM25 Search (no embedding cost!)
        status.write("🔎 **Step 2/3**: BM25 키워드 검색 중...")
        search_results = await agent.search_client.search(hypothetical_claim, top_k=5)
        status.write(f"✅ {len(search_results)}개 유사 특허 발견")
        
        # Step 3: Grading
        status.write("📊 **Step 3/4**: 관련성 평가 중...")
        grading = await agent.grade_results(user_idea, search_results)
        status.write(f"✅ 평균 관련성 점수: {grading.average_score:.2f}")
        
        # Step 4: Critical Analysis
        status.write("🧠 **Step 4/4**: 심층 분석 (All Elements Rule) 적용 중...")
        analysis = await agent.critical_analysis(user_idea, search_results)
        status.write("✅ 분석 완료!")
        
        status.update(label="✅ 분석 완료!", state="complete", expanded=False)
    
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
            }
            for r in search_results
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
            "conclusion": analysis.conclusion,
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    return result


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("# 🛡️ Patent Guard")
    st.markdown("### v2.0 - Self-RAG Engine")
    st.divider()
    
    # API Status
    if OPENAI_API_KEY:
        st.success("✅ OpenAI API 연결됨")
    else:
        st.error("❌ OpenAI API 키 없음")
        st.info("`.env` 파일에 `OPENAI_API_KEY`를 설정하세요.")
    
    st.divider()
    
    # Analysis History
    st.markdown("### 📜 분석 히스토리")
    if st.session_state.analysis_history:
        for i, hist in enumerate(reversed(st.session_state.analysis_history[-5:])):
            with st.expander(f"#{len(st.session_state.analysis_history)-i}: {hist['user_idea'][:20]}..."):
                risk = hist.get('analysis', {}).get('infringement', {}).get('risk_level', 'unknown')
                score = hist.get('analysis', {}).get('similarity', {}).get('score', 0)
                st.write(f"🎯 유사도: {score}/100")
                st.write(f"⚠️ 리스크: {risk.upper()}")
                st.write(f"🕐 {hist.get('timestamp', 'N/A')[:10]}")
    else:
        st.caption("아직 분석 기록이 없습니다.")
    
    st.divider()
    
    # API Usage Guide
    st.markdown("### 💰 API 비용 가이드")
    st.caption("""
    **분석 1회 예상 비용**: ~$0.01-0.03
    
    - HyDE: gpt-4o-mini
    - Search: BM25 (무료!)
    - Grading: gpt-4o-mini
    - Analysis: gpt-4o
    """)
    
    st.divider()
    st.markdown("##### Made by 뀨💕")


# =============================================================================
# Main Content
# =============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🛡️ Patent Guard v2.0</h1>
    <p style="font-size: 1.2rem; color: #888;">AI 기반 특허 선행 기술 조사 시스템</p>
    <p style="font-size: 0.9rem; color: #666;">Self-RAG | HyDE | All Elements Rule</p>
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

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyze_button = st.button(
        "🔍 특허 분석 시작",
        type="primary",
        use_container_width=True,
        disabled=not user_idea or not OPENAI_API_KEY,
    )

# Analysis Execution
if analyze_button and user_idea:
    status_container = st.container()
    
    try:
        # Run async analysis
        result = asyncio.run(run_analysis(user_idea, status_container))
        
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
    tab1, tab2, tab3, tab4 = st.tabs(["📝 종합 리포트", "🎯 유사도 분석", "⚠️ 침해 리스크", "🛡️ 회피 전략"])
    
    with tab1:
        st.markdown("### 📌 결론")
        st.info(analysis.get("conclusion", "분석 결과가 없습니다."))
        
        # Download button
        md_content = format_analysis_markdown(result)
        st.download_button(
            label="📥 리포트 다운로드 (Markdown)",
            data=md_content,
            file_name=f"patent_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
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
    
    # Referenced Patents
    st.divider()
    st.markdown("### 📚 참조된 선행 특허")
    
    for patent in result.get("search_results", []):
        with st.expander(f"📄 {patent.get('patent_id')} - Score: {patent.get('grading_score', 0):.2f}"):
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
    <p>🛡️ Patent Guard v2.0 | Self-RAG Patent Analysis System</p>
    <p style="font-size: 0.8rem;">Made with ❤️ by 뀨💕 | SKN22-3rd-2Team</p>
</div>
""", unsafe_allow_html=True)
