# 🏗️ 시스템 아키텍처

> **Patent Guard v2.0 - AI 특허 선행 기술 조사 시스템**  
> 작성일: 2026-01-27

---

## 1. 시스템 개요

### 1.1 목적
사용자가 특허 출원을 고려하는 아이디어를 입력하면, AI가 기존 특허 데이터베이스를 검색하여 **유사 특허**, **침해 리스크**, **회피 전략**을 제공하는 시스템

### 1.2 핵심 기술
| 기술 | 설명 |
|------|------|
| **Self-RAG** | 검색 결과를 비판적으로 평가하고 재검색하는 지능형 RAG |
| **HyDE** | 가상 문서 생성으로 검색 품질 향상 |
| **Chain-of-Thought** | 단계별 추론으로 정확한 분석 제공 |

---

## 2. 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Patent Guard v2.0                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │
│  │   사용자    │───▶│  Patent     │───▶│    OpenAI API           │  │
│  │   입력      │    │  Agent      │    │  ┌─────────────────────┐│  │
│  │             │    │             │    │  │ text-embedding-3    ││  │
│  │ "아이디어"  │    │ ┌─────────┐ │    │  │ gpt-4o-mini         ││  │
│  └─────────────┘    │ │  HyDE   │ │    │  │ gpt-4o              ││  │
│                     │ └─────────┘ │    │  └─────────────────────┘│  │
│                     │ ┌─────────┐ │    └─────────────────────────┘  │
│                     │ │ Grading │ │                                 │
│                     │ └─────────┘ │    ┌─────────────────────────┐  │
│                     │ ┌─────────┐ │    │    특허 데이터           │  │
│                     │ │  CoT    │ │───▶│  ┌─────────────────────┐│  │
│                     │ │Analysis │ │    │  │ 10,000 Patents      ││  │
│                     │ └─────────┘ │    │  │ (processed_patents  ││  │
│                     └─────────────┘    │  │  _10k.json)         ││  │
│                            │           │  └─────────────────────┘│  │
│                            ▼           └─────────────────────────┘  │
│                     ┌─────────────┐                                 │
│                     │  분석 결과  │                                 │
│                     │ ┌─────────┐ │                                 │
│                     │ │유사도   │ │                                 │
│                     │ │침해위험 │ │                                 │
│                     │ │회피전략 │ │                                 │
│                     │ └─────────┘ │                                 │
│                     └─────────────┘                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 핵심 컴포넌트

### 3.1 Patent Agent (`patent_agent.py`)

메인 분석 엔진으로 3단계 Self-RAG 파이프라인 구현

```
┌──────────────────────────────────────────────────────────────┐
│                      Patent Agent                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  [Stage 1] HyDE (Hypothetical Document Embedding)            │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ 사용자 아이디어 → GPT-4o-mini → 가상 특허 청구항       │  │
│  │ → text-embedding-3-small → 벡터 검색                   │  │
│  └────────────────────────────────────────────────────────┘  │
│                           ↓                                  │
│  [Stage 2] Grading & Rewrite Loop                            │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ 검색 결과 5개 → GPT-4o-mini → 관련성 점수 (0~1)        │  │
│  │ 평균 < 0.6 → 쿼리 재작성 → 재검색 (1회)                │  │
│  └────────────────────────────────────────────────────────┘  │
│                           ↓                                  │
│  [Stage 3] Critical CoT Analysis                             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ 최종 선정 특허 → GPT-4o → 상세 분석                    │  │
│  │ [유사도] + [침해 리스크] + [회피 전략]                  │  │
│  │ 각 분석마다 근거 특허 번호 명시                        │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 데이터 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Pipeline                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [1] BigQuery Extractor                                         │
│      Google Patents → SQL 쿼리 → 원본 JSON                      │
│                                                                 │
│  [2] Preprocessor                                               │
│      원본 JSON → 청구항 파싱 → 청킹 → 전처리 JSON               │
│                                                                 │
│  [3] Mock Milvus Client                                         │
│      전처리 JSON → 랜덤 샘플링 (데모용)                         │
│      → [미래] 실제 Milvus 벡터 DB 연동                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 기술 스택

### 4.1 Backend

| 구분 | 기술 |
|------|------|
| **언어** | Python 3.11 |
| **프레임워크** | asyncio (비동기) |
| **데이터 검증** | Pydantic |
| **환경 변수** | python-dotenv |

### 4.2 AI/ML

| 구분 | 기술 |
|------|------|
| **LLM** | OpenAI GPT-4o, GPT-4o-mini |
| **Embedding** | OpenAI text-embedding-3-small |
| **Vector DB** | Milvus (계획) / Mock (현재) |

### 4.3 Data

| 구분 | 기술 |
|------|------|
| **Data Source** | Google Patents (BigQuery) |
| **Storage** | JSON 파일 |
| **Query Language** | SQL |

---

## 5. API 설계

### 5.1 OpenAI API 사용

| 모델 | 용도 | 비용 |
|------|------|------|
| `text-embedding-3-small` | 텍스트 임베딩 | $0.02/1M tokens |
| `gpt-4o-mini` | HyDE, Grading | $0.15/1M input |
| `gpt-4o` | Critical Analysis | $5.00/1M input |

### 5.2 응답 형식 (JSON Mode)

```python
# Grading Response
{
  "results": [
    {"patent_id": "US-123", "score": 0.8, "reason": "..."}
  ],
  "average_score": 0.75
}

# Analysis Response
{
  "similarity": {"score": 45, "summary": "...", "evidence": [...]},
  "infringement": {"risk_level": "medium", ...},
  "avoidance": {"strategies": [...], ...},
  "conclusion": "..."
}
```

---

## 6. 파일 구조

```
src/
├── patent_agent.py       # 🎯 메인 분석 에이전트
│   ├── PatentAgent       # Self-RAG 파이프라인
│   ├── MockMilvusClient  # 벡터 검색 (Mock)
│   └── Pydantic Models   # 응답 구조 정의
│
├── bigquery_extractor.py # 📥 BigQuery 데이터 추출
│   ├── SQLGenerator      # 동적 SQL 생성
│   └── BigQueryExtractor # 쿼리 실행 & 추출
│
├── preprocessor.py       # 🔧 데이터 전처리
│   ├── PatentPreprocessor
│   └── ClaimParser       # 청구항 파싱
│
├── pipeline.py           # ⚙️ 파이프라인 오케스트레이터
│   ├── stage_1_extraction
│   ├── stage_2_preprocessing
│   └── run_full_pipeline
│
├── config.py             # ⚙️ 설정 관리
│   ├── BigQueryConfig
│   ├── DomainConfig
│   └── EmbeddingConfig
│
├── vector_db.py          # 🗄️ Milvus 연동 (계획)
└── embedder.py           # 🧠 임베딩 생성 (OpenAI)
```

---

## 7. 데이터 흐름

```
[사용자 입력]
    │
    ▼
[HyDE] ─────────────────────────────────────────────┐
    │ 가상 청구항 생성 (GPT-4o-mini)                │
    ▼                                               │
[Embedding] ────────────────────────────────────────┤
    │ text-embedding-3-small                        │
    ▼                                               │
[Vector Search] ────────────────────────────────────┤
    │ 10K 특허 중 Top-5 검색                        │
    ▼                                               │
[Grading] ──────────────────────────────────────────┤
    │ 관련성 평가 (GPT-4o-mini)                     │
    │ 점수 < 0.6 → 재검색                           │
    ▼                                               │
[Critical Analysis] ────────────────────────────────┤
    │ 상세 분석 (GPT-4o)                            │
    ▼                                               │
[Output]                                            │
    ├─ 유사도 평가 (0-100)                          │
    ├─ 침해 리스크 (high/medium/low)                │
    └─ 회피 전략                                    │
                                                    │
    [Total: ~5-10 API calls per analysis]───────────┘
```

---

## 8. 검색 엔진

### 8.1 현재: BM25 키워드 검색 (무료)

```
[사용자 아이디어]
     ↓
[HyDE] 가상 청구항 생성 (GPT-4o-mini)
     ↓
[BM25] 키워드 매칭 기반 검색 (rank_bm25)
     ↓
[Top-5 특허 반환]
```

**장점:**
- ✅ API 비용 없음 (로컬 처리)
- ✅ 빠른 검색 속도
- ✅ 데모/프로토타입에 적합

**단점:**
- ⚠️ 동의어/의미 유사성 인식 불가
- ⚠️ 70-80% 검색 품질

### 8.2 향후: 벡터 임베딩 검색 (권장)

| 단계 | 작업 | 비용 |
|------|------|------|
| 1 | 10K 특허 임베딩 생성 | ~$0.20 (1회) |
| 2 | 임베딩 저장 (`.npy` 파일) | - |
| 3 | 쿼리 시 코사인 유사도 계산 | ~$0.002/쿼리 |

```python
# 향후 업그레이드 코드 예시
from scipy.spatial.distance import cosine

# 1회 전처리
embeddings = np.load("patent_embeddings.npy")

# 검색
query_emb = openai.embed(hypothetical_claim)
similarities = [1 - cosine(query_emb, p_emb) for p_emb in embeddings]
top_5 = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:5]
```

**품질 향상:**
- BM25: 70-80% → **임베딩: 90-95%**

---

## 9. 확장 계획

### 9.1 Demo → Production

| 현재 (Demo) | 향후 (Production) |
|-------------|-------------------|
| BM25 키워드 검색 | **벡터 임베딩 검색** |
| 10K 특허 | On-demand BigQuery 검색 |
| JSON 저장 | Database (PostgreSQL) |
| Streamlit | FastAPI + React |

### 9.2 추가 기능

- [x] 웹 UI (Streamlit) ✅
- [ ] 벡터 임베딩 검색 업그레이드
- [ ] 실시간 BigQuery 검색
- [ ] 분석 결과 히스토리 DB
- [ ] 다국어 지원 (한/영/중/일)

---

## 9. 환경 변수

```env
# 필수
OPENAI_API_KEY=your-api-key
GCP_PROJECT_ID=your-project-id

# 선택 (기본값 있음)
EMBEDDING_MODEL=text-embedding-3-small
GRADING_MODEL=gpt-4o-mini
ANALYSIS_MODEL=gpt-4o
GRADING_THRESHOLD=0.6
TOP_K_RESULTS=5
```

---

*작성: Patent Guard Team - 뀨💕*
