"""
Core analysis logic orchestration.
"""
import time
import asyncio
import streamlit as st
from datetime import datetime
from src.patent_agent import PatentAgent, PatentSearchResult

async def run_analysis_streaming(agent, user_idea: str, results, output_container):
    """Run streaming analysis and display in real-time."""
    full_text = ""
    placeholder = output_container.empty()
    
    async for token in agent.critical_analysis_stream(user_idea, results):
        full_text += token
        placeholder.markdown(full_text + "▌")  # Cursor effect
    
    placeholder.markdown(full_text)  # Final output without cursor
    return full_text


async def run_full_analysis(user_idea: str, status_container, streaming_container, db_client, use_hybrid: bool = True):
    """Run the complete patent analysis with streaming."""
    
    # Create agent with cached DB client
    agent = PatentAgent(db_client=db_client)
    
    results = []
    start_time = time.time()
    
    # Progress bar
    progress_bar = status_container.progress(0, text="🚀 분석 시작...")
    
    with status_container.status("🔍 특허 분석 중...", expanded=True) as status:
        # Step 1: HyDE (~3초)
        progress_bar.progress(5, text="📝 Step 1/4: 가상 청구항 생성 중... (예상: 3초)")
        status.write("📝 **Step 1/4**: HyDE - 가상 청구항 생성 중...")
        hypothetical_claim = await agent.generate_hypothetical_claim(user_idea)
        progress_bar.progress(25, text="✅ Step 1 완료!")
        status.write(f"✅ 가상 청구항 생성 완료")
        status.write(f"```\n{hypothetical_claim[:200]}...\n```")
        
        # Step 2: Hybrid Search (~2초)
        search_type = "Hybrid (Dense + BM25)" if use_hybrid else "Dense Only"
        progress_bar.progress(30, text=f"🔎 Step 2/4: {search_type} 검색 중... (예상: 2초)")
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
        
        progress_bar.progress(50, text="✅ Step 2 완료!")
        status.write(f"✅ {len(results)}개 유사 특허 발견")
        
        # Step 3: Grading (~3초)
        progress_bar.progress(55, text="📊 Step 3/4: 관련성 평가 중... (예상: 3초)")
        status.write("📊 **Step 3/4**: 관련성 평가 중...")
        grading = await agent.grade_results(user_idea, results)
        progress_bar.progress(75, text="✅ Step 3 완료!")
        status.write(f"✅ 평균 관련성 점수: {grading.average_score:.2f}")
        
        status.update(label="✅ 검색 완료! 분석 스트리밍 시작...", state="complete", expanded=False)
    
    # Step 4: Streaming Analysis (~10초)
    progress_bar.progress(80, text="🧠 Step 4/4: AI 분석 스트리밍 중... (예상: 10초)")
    streaming_container.markdown("### 🧠 실시간 분석 결과")
    streaming_container.caption("AI가 분석 내용을 실시간으로 생성합니다...")
    
    streamed_text = await run_analysis_streaming(agent, user_idea, results, streaming_container)
    
    # Also get structured analysis for result storage
    analysis = await agent.critical_analysis(user_idea, results)
    
    # Complete progress bar
    elapsed = time.time() - start_time
    progress_bar.progress(100, text=f"✅ 분석 완료! (소요 시간: {elapsed:.1f}초)")
    
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
