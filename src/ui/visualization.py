"""
Visualization module for Patent Landscape Map.
Effectively visualizes the relationship between User Idea and Search Results.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

def render_patent_map(result: dict):
    """
    Render a premium interactive Patent Landscape Map.
    
    Features:
    - Quadrant analysis (Direct Risk, Technical Reference, Noise, Strategic Avoidance)
    - User Idea Center Point
    - Distance-based conceptual alignment visualization
    """
    search_results = result.get('search_results', [])
    user_idea = result.get('user_idea', '내 아이디어')
    
    if not search_results:
        st.caption("시각화할 데이터가 충분하지 않습니다.")
        return

    # Prepare data for DataFrame
    data = []
    
    # 1. Add User Idea as the Origin/Goal point
    data.append({
        "Patent ID": "🎯 My Idea",
        "Title": "내 아이디어 (분석 기준점)",
        "Conceptual Alignment": 1.0,  # Center point for idea
        "Analytical Depth": 1.0,
        "Relevance": 15,
        "Category": "My Idea",
        "Abstract": user_idea[:200]
    })
    
    # 2. Add search results
    for r in search_results:
        # We use grading_score for alignment and score/stats for depth
        alignment = r.get('grading_score', 0.5)
        # Combine dense and sparse scores for depth (normalized roughly)
        depth = (r.get('dense_score', 0) * 0.7 + min(r.get('sparse_score', 0) / 50, 1.0) * 0.3)
        
        grade = r.get('grading_score', 0)
        
        if grade >= 0.75:
            cat = "🚨 침해 주의 (高)"
        elif grade >= 0.5:
            cat = "🟡 기술적 참고"
        elif alignment > 0.6 and depth < 0.4:
            cat = "🕵️ 숨겨진 경쟁자"
        else:
            cat = "📗 단순 키워드 중복"
            
        data.append({
            "Patent ID": r.get('patent_id'),
            "Title": r.get('title'),
            "Conceptual Alignment": alignment,
            "Analytical Depth": depth,
            "Relevance": grade * 25 + 5,
            "Category": cat,
            "Abstract": r.get('abstract', '')[:150] + "..."
        })
        
    df = pd.DataFrame(data)
    
    # Create Scatter Plot with premium styling
    fig = px.scatter(
        df,
        x="Conceptual Alignment",
        y="Analytical Depth",
        size="Relevance",
        color="Category",
        hover_name="Title",
        hover_data={"Patent ID": True, "Abstract": True, "Relevance": False},
        color_discrete_map={
            "My Idea": "#00d4ff",
            "🚨 침해 주의 (高)": "#ff4b4b",
            "🟡 기술적 참고": "#ffa500",
            "🕵️ 숨겨진 경쟁자": "#6c5ce7",
            "📗 단순 키워드 중복": "#a0a0a0"
        },
        title="✨ Premium Patent Landscape Analysis",
        template="plotly_dark"
    )
    
    # Add Quadrant Backgrounds/Annotations using shapes if possible, or just layout lines
    fig.add_hline(y=0.5, line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.2)")
    fig.add_vline(x=0.5, line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.2)")
    
    fig.update_layout(
        xaxis_title="🎯 기술적 정렬도 (Conceptual Alignment)",
        yaxis_title="🔍 분석 심도 (Analytical Depth)",
        legend_title="Risk & Value",
        hovermode="closest",
        height=600,
        margin=dict(l=60, r=60, t=100, b=60),
        plot_bgcolor="rgba(15,17,23,1)",
        paper_bgcolor="rgba(15,17,23,1)",
        xaxis=dict(range=[-0.1, 1.1], gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(range=[-0.1, 1.1], gridcolor="rgba(255,255,255,0.05)"),
        font=dict(family="Pretendard, sans-serif", size=13)
    )
    
    # Add Quadrant Labels
    fig.add_annotation(x=0.85, y=0.9, text="<b>HIGH RISK ZONE</b>", showarrow=False, font=dict(color="#ff4b4b", size=14))
    fig.add_annotation(x=0.15, y=0.9, text="Keyword Noise", showarrow=False, font=dict(color="#a0a0a0"))
    fig.add_annotation(x=0.85, y=0.1, text="Conceptual Competitors", showarrow=False, font=dict(color="#6c5ce7"))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Premium guide
    st.info("""
    💡 **분석 가이드**:
    - **중앙(🎯)**: 당신의 아이디어입니다. 가까울수록 실질적인 경쟁/침해 리스크가 높습니다.
    - **우측 상단**: 키워드와 핵심 원리가 모두 유사한 **직적적 침해 위협** 영역입니다.
    - **우측 하단**: 키워드는 다르지만 기술적 사상이 유사한 **잠재적 경쟁자**입니다. 회피 설계가 필요할 수 있습니다.
    """)
 