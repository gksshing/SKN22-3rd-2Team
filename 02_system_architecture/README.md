# 🏗️ 시스템 아키텍처

> **⚡ 쇼특허 (Short-Cut) - AI 특허 선행 기술 조사 시스템**  
> Team: 뀨💕 | 작성일: 2026-01-30

---

## 1. 시스템 개요

### 1.1 목적
사용자가 특허 출원을 고려하는 아이디어를 입력하면, AI가 기존 특허 데이터베이스를 검색하여 **유사 특허**, **침해 리스크**, **회피 전략**을 제공하는 시스템입니다.

### 1.2 핵심 기술 (v3.0)

| 기술 | 설명 |
|------|------|
| **Self-RAG** | 검색 결과의 관련성을 평가(Grading)하고 필요 시 쿼리를 재작성(Rewrite)하여 재검색하는 지능형 루프 |
| **HyDE** | 사용자 아이디어로부터 '가상 청구항'을 생성하여 검색의 정확도(특히 재현율)를 향상 |
| **Multi-Query** | 아이디어를 기술적 관점, 청구항 관점, 문제해결 관점 등 다각도로 확장하여 검색 |
| **Hybrid Search** | **Pinecone Serverless** (Dense + Sparse) Unified Index (No Local DB) |
| **Reranker** | Cross-Encoder 모델(`ms-marco-MiniLM`)을 사용하여 검색 결과의 순위를 정밀하게 재조정 |
| **LLM Streaming** | 분석 과정을 실시간으로 스트리밍하여 사용자의 체감 대기 시간을 최소화 |

---

## 2. 전체 아키텍처 (System Pipeline)

데이터의 흐름과 주요 컴포넌트 간의 상호작용을 나타냅니다.

```mermaid
graph TD
    User[사용자 입력] --> UI[Streamlit App (app.py)]
    UI --> Logic[Analysis Logic (analysis_logic.py)]
    
    subgraph "Patent Agent (Self-RAG)"
        Logic --> Agent[PatentAgent]
        Agent --> HyDE[HyDE: 가상 청구항 생성]
        HyDE --> MultiQuery[Multi-Query 생성]
        MultiQuery --> Search[Hybrid Search]
        
        Search --> Dense[Pinecone (Dense Vector)]
        Search --> Sparse[Pinecone (Sparse Vector)]
        
        Dense & Sparse --> Mixed[Weighted Sum Scoring]
        Mixed --> Rerank[Cross-Encoder Reranker]
        
        Rerank --> Grade{Grading (관련성 평가)}
        Grade -- "Low Score (<0.6)" --> Rewrite[Query Rewrite]
        Rewrite --> Search
        Grade -- "High Score" --> Analysis[Critical CoT Analysis]
    end
    
    Analysis --> Stream[Streaming Output]
    Stream --> UI
```

---

## 3. 핵심 컴포넌트 상세

### 3.1 Patent Agent (`src/patent_agent.py`)
시스템의 두뇌 역할을 하는 에이전트 클래스입니다.
- **HyDE (Hypothetical Document Embedding)**: `gpt-4o-mini`를 사용하여 아이디어를 특허 청구항 스타일로 변환합니다.
- **Search Loop**: 초기 검색 결과가 만족스럽지 않을 경우(Grading 점수 미달), 검색 쿼리를 스스로 수정하여 재검색을 수행합니다.
- **Critical Analysis**: `gpt-4o`를 사용하여 검색된 특허와 사용자 아이디어를 '청구항 단위'로 정밀 비교 분석합니다. (유사도, 침해 리스크, 회피 전략)

### 3.2 Hybrid Search & Reranker
검색의 재현율(Recall)과 정밀도(Precision)를 모두 잡기 위한 전략입니다.
- **Dense Search (Pinecone)**: 문맥적 의미 유사성을 기반으로 검색합니다. (Model: `text-embedding-3-small`)
- **Sparse Search (BM25)**: 키워드 매칭(TF-IDF 변형)을 기반으로 검색합니다. (전문 용어 검색에 강점)
- **RRF Fusion**: 두 검색 결과의 순위를 상호보완적으로 결합합니다.
- **Reranker**: `Cross-Encoder`를 사용하여 상위 결과들의 문맥적 연관성을 다시 한 번 정밀하게 채점하여 최종 순위를 결정합니다.

---

## 4. 데이터베이스 및 인프라

### 4.1 Vector Database (Pinecone)
- **Type**: Serverless Index (AWS/GCP)
- **Dimension**: 1536 (OpenAI Embedding)
- **Metric**: DotProduct (Required for Sparse-Dense Hybrid)
- **Metadata**: 특허 번호, 제목, 초록, 청구항, IPC 코드 등 저장 (Unified Storage)

### 4.2 Sparse Encoder
- **Library**: `pinecone-text`
- **Role**: 텍스트를 Sparse Vector로 변환 (Stateless Client-side logic)
- **Config**: `data/index/bm25_params.json` (Lightweight Params)
- **Note**: 기존 `rank_bm25` 로컬 인덱스 파일(.pkl)은 완전히 제거됨.

---

## 5. 분석 프로세스 (Logic Flow)

1. **User Input**: 사용자가 아이디어를 입력합니다.
2. **HyDE**: "이 아이디어가 특허로 출원된다면 어떤 청구항일까?"를 상상하여 가상 문서를 생성합니다.
3. **Retrieval**: 가상 문서와 원본 아이디어를 바탕으로 3가지 관점의 쿼리를 생성하고, Hybrid Search를 수행합니다.
4. **Reranking**: 검색된 후보군 중 상위 문서들을 정밀 모델로 재정렬합니다.
5. **Grading**: 상위 5개 문서가 실제로 관련이 있는지 LLM이 채점합니다. (관련성이 낮으면 쿼리 수정 후 3번으로 복귀)
6. **Analysis**: 최종 선정된 특허들과 아이디어를 비교하여 유사도, 기술적 차이점, 침해 가능성 등을 심층 분석합니다.
7. **Streaming**: 분석되는 내용을 실시간으로 화면에 출력합니다.


*작성: Team 뀨💕*
