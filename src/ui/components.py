"""
UI Components for the application.
"""
import streamlit as st
from datetime import datetime
from src.utils import get_risk_color, get_score_color, get_patent_link, display_patent_with_link, format_analysis_markdown
from src.ui.styles import apply_theme_css

def render_header():
    """Render the application header."""
    st.markdown("""
    <div class="main-header">
        <h1>⚡ 쇼특허 (Short-Cut) v3.0</h1>
        <p style="font-size: 1.2rem; color: #888;">AI 기반 특허 선행 기술 조사 시스템</p>
        <p style="font-size: 0.9rem; color: #666;">Self-RAG | Hybrid Search | LLM Streaming</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar(openai_api_key, db_client, db_stats):
    """Render the sidebar."""
    with st.sidebar:
        st.markdown("# ⚡ 쇼특허")
        st.markdown("### Short-Cut v3.0")
        st.divider()
        
        # Settings - Theme
        st.markdown("### 🎨 테마 설정")
        theme = st.radio(
            "테마 선택",
            ["🌙 다크 모드", "☀️ 라이트 모드"],
            index=0,
            horizontal=True,
            label_visibility="collapsed"
        )
        is_dark_mode = theme == "🌙 다크 모드"
        
        # Apply theme CSS
        apply_theme_css(is_dark_mode)
        
        st.divider()
        
        # System Status
        st.markdown("### ⚡ System Status")
        
        # API Status
        if openai_api_key:
            st.success("✅ OpenAI API 연결됨")
        else:
            st.error("❌ OpenAI API 키 없음")
            st.info("`.env` 파일에 `OPENAI_API_KEY`를 설정하세요.")
        
        # DB Index Status
        if db_client:
            st.success(f"✅ Hybrid 인덱스 로드됨")
            st.caption(f"   🌲 Pinecone: Connected")
            if db_stats.get('bm25_initialized'):
                st.caption(f"   📝 BM25 (Local): {db_stats.get('bm25_docs', 0):,}개 문서")
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
            
            # Using absolute import for session manager in component might be cleaner if passed as arg or callback
            # But currently sticking to app logic, session state modification should work.
            if st.button("🗑️ 기록 삭제", use_container_width=True):
                # This should be handled by a callback or clearing session state here
                st.session_state.analysis_history = []
                # Ideally, clear persistent history too via session manager
                # For now, we assume the caller handles or we trigger rerun
                # But components should avoid side-effects like reruns if possible. 
                # Let's keep the button here but note that app.py might need to handle the action if complex.
                # Actually, implementing the action here using session_state is fine.
                from src.session_manager import clear_user_history
                clear_user_history()
        
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
        
        # User Info (Debug)
        user_id = st.session_state.get("user_id", "unknown")
        st.caption(f"👤 User ID: `{user_id}`")
        st.markdown("##### Team 뀨💕")
        
        return use_hybrid


def render_search_results(result):
    """Render search result metrics and details."""
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
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📝 종합 리포트", "🗺️ 특허 지형도", "🎯 유사도 분석", "⚠️ 침해 리스크", "🛡️ 회피 전략", "🔬 구성요소 대비"])
    
    with tab1:
        st.markdown("### 📌 결론")
        conclusion_text = analysis.get("conclusion", "분석 결과가 없습니다.")
        st.info(conclusion_text)
        
        # Downloads
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            md_content = format_analysis_markdown(result)
            st.download_button(
                label="📥 리포트 다운로드 (Markdown)",
                data=md_content,
                file_name=f"shortcut_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )
            
        with col_d2:
            if st.button("📄 리포트 다운로드 (PDF)"):
                with st.spinner("PDF 생성 중..."):
                    try:
                        from src.pdf_generator import PDFGenerator
                        import tempfile
                        
                        pdf_gen = PDFGenerator()
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            pdf_path = pdf_gen.generate_report(result, tmp.name)
                            
                            with open(pdf_path, "rb") as f:
                                pdf_data = f.read()
                                
                            st.download_button(
                                label="📥 PDF 저장하기",
                                data=pdf_data,
                                file_name=f"shortcut_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                            )
                    except Exception as e:
                        st.error(f"PDF 생성 실패: {e}")

    with tab2:
        from src.ui.visualization import render_patent_map
        render_patent_map(result)
    
    with tab3:
        similarity = analysis.get("similarity", {})
        st.markdown(f"### 유사도 점수: {similarity.get('score', 0)}/100")
        st.markdown(f"**분석 요약**: {similarity.get('summary', 'N/A')}")
        
        st.markdown("**공통 기술 요소:**")
        for elem in similarity.get("common_elements", []):
            st.markdown(f"- {elem}")
        
        st.markdown("**근거 특허:**")
        for patent in similarity.get("evidence", []):
            display_patent_with_link(patent)
    
    with tab4:
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
            display_patent_with_link(patent)
            
    with tab5:
        avoidance = analysis.get("avoidance", {})
        st.markdown(f"**권장 전략**: {avoidance.get('summary', 'N/A')}")
        
        st.markdown("**회피 설계 방안:**")
        for strategy in avoidance.get("strategies", []):
            st.markdown(f"- ✅ {strategy}")
        
        st.markdown("**대안 기술:**")
        for alt in avoidance.get("alternatives", []):
            st.markdown(f"- 💡 {alt}")
            
    with tab6:
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
