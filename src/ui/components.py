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
    
    # Log status to terminal (not shown in UI)
    import logging
    logger = logging.getLogger(__name__)
    
    if openai_api_key:
        logger.info("✅ OpenAI API 연결됨")
    else:
        logger.warning("❌ OpenAI API 키 없음")
        
    if db_client:
        logger.info(f"✅ Hybrid 인덱스 로드됨 - Pinecone Connected")
        if db_stats.get('bm25_initialized'):
            logger.info(f"   📝 BM25 (Local): {db_stats.get('bm25_docs', 0):,}개 문서")
    else:
        logger.warning("⚠️ DB 연결 실패")
    
    with st.sidebar:
        st.markdown("# ⚡ 쇼특허")
        st.markdown("### Short-Cut v3.0")
        st.divider()
        
        # Apply theme CSS (Hardcoded Ivory/Light)
        apply_theme_css()
        
        # ----------------------------------------------------
        # 🔧 검색 옵션 (IPC 필터링)
        # ----------------------------------------------------
        st.markdown("### 🔧 검색 옵션")
        
        IPC_CATEGORIES = {
            "G06 (컴퓨터/AI)": "G06",
            "H04 (통신/네트워크)": "H04",
            "A61 (의료/헬스케어)": "A61",
            "H01 (반도체/전자)": "H01",
            "B60 (차량/운송)": "B60",
            "C12 (바이오/생명)": "C12",
            "F02 (기계/엔진)": "F02",
        }
        
        selected_categories = st.multiselect(
            "관심 기술 분야 (선택 시 필터링)",
            options=list(IPC_CATEGORIES.keys()),
            default=[],
            help="특정 기술 분야(IPC)로 검색 범위를 제한하여 정확도를 높입니다."
        )
        
        selected_ipc_codes = [IPC_CATEGORIES[cat] for cat in selected_categories]
        
        st.divider()
        
        # ----------------------------------------------------
        # 📖 특허 가이드 (Patent Guide) - YouTube Popup
        # ----------------------------------------------------
        st.markdown("### 📖 특허 가이드")
        st.caption("처음 사용하시나요? 가이드 영상을 확인하세요.")
        
        @st.dialog("📖 특허 출원 가이드", width="large")
        def show_patent_guide_popup():
            st.write("**특허 출원 전 알아야 할 핵심 정보:**")
            
            # YouTube video (can be changed to relevant guide video)
            video_url = "https://www.youtube.com/watch?v=HSWXcMSneB4"
            st.video(video_url)
            
            st.write("---")
            st.caption("닫기 버튼이나 배경을 클릭하면 팝업이 닫힙니다.")
        
        if st.button("🎥 가이드 영상 보기", use_container_width=True):
            show_patent_guide_popup()
        
        st.divider()
        
        # ----------------------------------------------------
        # 📜 분석 히스토리
        # ----------------------------------------------------
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
            
        if st.button("🗑️ 기록 삭제", use_container_width=True):
            st.session_state.analysis_history = []
            from src.session_manager import clear_user_history
            clear_user_history()
        
        st.divider()
        
        # Team Info
        st.markdown("##### Team 뀨💕")
        
        # Return tuple: (use_hybrid=True always, selected_ipc_codes)
        return True, selected_ipc_codes


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
        
        # ========================================
        # Feedback Section
        # ========================================
        st.divider()
        st.markdown("### 📣 분석 품질 피드백")
        st.caption("이 분석 결과가 도움이 되었나요? 피드백을 남겨주시면 검색 품질 개선에 활용됩니다.")
        
        from src.feedback_logger import save_feedback
        
        user_idea = result.get("user_idea", "")
        search_results = result.get("search_results", [])
        user_id = st.session_state.get("user_id", "unknown")
        
        if search_results:
            for i, patent in enumerate(search_results[:5]):  # Top 5 patents
                patent_id = patent.get("patent_id", f"unknown_{i}")
                title = patent.get("title", "제목 없음")[:50]
                grading_score = patent.get("grading_score", 0)
                
                col1, col2, col3 = st.columns([4, 1, 1])
                
                with col1:
                    st.markdown(f"**{i+1}. {title}...** (유사도: {grading_score:.0%})")
                
                with col2:
                    if st.button("👍", key=f"fb_pos_{patent_id}_{i}", help="이 특허는 관련 있어요"):
                        save_feedback(
                            query=user_idea,
                            patent_id=patent_id,
                            score=1,
                            user_id=user_id,
                            metadata={"grading_score": grading_score, "title": title}
                        )
                        st.toast(f"✅ '{patent_id}' 관련성 피드백 저장됨!")
                
                with col3:
                    if st.button("👎", key=f"fb_neg_{patent_id}_{i}", help="이 특허는 관련 없어요"):
                        save_feedback(
                            query=user_idea,
                            patent_id=patent_id,
                            score=-1,
                            user_id=user_id,
                            metadata={"grading_score": grading_score, "title": title}
                        )
                        st.toast(f"❌ '{patent_id}' 비관련 피드백 저장됨!")

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
