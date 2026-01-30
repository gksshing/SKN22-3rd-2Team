"""
Short-Cut v3.0 Main Application (Simple Download Version).
"""
import asyncio
import os
import streamlit as st
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

# Streamlit Config (반드시 맨 처음에 위치해야 함)
st.set_page_config(
    page_title="Short-Cut v3.0",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Imports
from src.session_manager import init_session_state, load_history, save_result_to_history
from src.ui.styles import get_main_css
from src.ui.components import render_header, render_sidebar, render_search_results
from src.analysis_logic import run_full_analysis

# Initialize Session & History
init_session_state()
load_history()

# Apply Global CSS
st.markdown(get_main_css(), unsafe_allow_html=True)

# Render Header
render_header()

# Cached Resource Loading
@st.cache_resource
def load_db_client():
    """Load Pinecone + BM25 hybrid client."""
    from src.vector_db import PineconeClient
    try:
        client = PineconeClient()
        db_stats = client.get_stats()
        return client, db_stats
    except Exception as e:
        print(f"DB Init failed: {e}")
        return None, {}

DB_CLIENT, DB_STATS = load_db_client()

# --- [사이드바 구성: 다운로드 버튼 배치] ---
with st.sidebar:
    st.markdown("### 🧭 메뉴")
    
    # 1. 홈 버튼 (필요 시 화면 초기화 용도)
    if st.button("🏠 분석 화면 초기화", use_container_width=True):
        st.session_state.current_result = None # 결과 초기화
        st.rerun()
        
    st.divider()
    
    # 2. [핵심] 용어 사전 다운로드 버튼
    # static 폴더 안의 파일 경로
    target_filename = "지식재산권용어사전_편집본_v16.pdf"
    file_path = os.path.join("static", target_filename)
    
    # 파일이 존재하는지 확인 후 버튼 생성
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            file_data = f.read()
            
        st.download_button(
            label="📥 지식재산권 용어 사전 다운로드",
            data=file_data,
            file_name=target_filename, # 다운로드될 때의 파일명
            mime="application/pdf",
            use_container_width=True
        )
    else:
        st.error("⚠️ 용어 사전 파일이 없습니다.")
        st.caption(f"경로: {file_path}")
    
    st.divider()
    
    # 기존 사이드바 설정 (API 키 등)
    use_hybrid = render_sidebar(OPENAI_API_KEY, DB_CLIENT, DB_STATS)


# --- [메인 콘텐츠: 특허 분석 단일 화면] ---

st.markdown("### 💡 아이디어 입력")
st.caption("특허로 출원하려는 아이디어를 설명해주세요. 유사 특허를 찾아 침해 리스크를 분석합니다.")

user_idea = st.text_area(
    label="아이디어 설명",
    placeholder="예: 딥러닝 기반 문서 요약 시스템으로, 긴 문서를 입력받아 핵심 내용을 추출하고 요약문을 생성합니다...",
    height=120,
    label_visibility="collapsed",
)

# Analysis Check
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
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            run_full_analysis(user_idea, status_container, streaming_container, DB_CLIENT, use_hybrid=use_hybrid)
        )
        
        loop.close()
        save_result_to_history(result)
            
    except Exception as e:
        st.error(f"❌ 분석 중 오류가 발생했습니다: {str(e)}")
        st.info("💡 OpenAI API 키를 확인하거나, 잠시 후 다시 시도해주세요.")

# Results Display
if st.session_state.current_result:
    render_search_results(st.session_state.current_result)