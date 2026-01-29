# ⚡ 쇼특허 (Short-Cut)

**AI 기반 특허 선행 기술 조사 시스템**

사용자의 아이디어를 입력하면 기존 특허와 비교하여 **유사도**, **침해 리스크**, **회피 전략**을 분석해주는 Self-RAG 기반 특허 분석 도구입니다.

> **Team 뀨💕** | [기술 제안서](report/v3_technical_proposal.md) | [기술 리포트](report/v3_technical_report.md)

---

## 🎯 주요 기능

| 기능 | 설명 |
|------|------|
| **Multi-Query RAG** | 3가지 관점(기술/청구항/문제해결)으로 쿼리를 확장하여 검색 커버리지 극대화 |
| **IPC Filtering** | 관심 기술 분야(IPC) 필터링으로 검색 정확도 향상 (User-friendly 라벨 제공) |
| **Hybrid Search** | Pinecone (Dense) + Local BM25 (Sparse) + RRF 융합 검색 |
| **Reranker** | Cross-Encoder(ms-marco)를 활용한 검색 결과 정밀 재정렬 |
| **Claim-Level Analysis** | '모든 구성요소 법칙'을 적용하여 각 특허의 위험 청구항 정밀 분석 |
| **Feedback Loop** | 사용자 피드백(👍/👎) 수집 및 Reranker 학습 데이터 구축 |
| **Serverless DB** | Pinecone 벡터 DB를 활용한 확장성 있는 데이터 관리 |
| **LLM Streaming** | 실시간 분석 결과 출력 (0초 체감 대기시간) |
| **Visualization** | 특허 지형도 (Jitter/Connection Line) 및 전략 가이드 제공 |

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
conda create -n patent-guard python=3.11 -y
conda activate patent-guard

# 의존성 설치 (sentence-transformers 포함)
pip install -r requirements.txt

# NLP 모델 다운로드 (선택)
python -m spacy download en_core_web_sm
```

### 2. 환경 변수 설정

```bash
cp .env.example .env
```

`.env` 파일 편집:
```env
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
GCP_PROJECT_ID=your-gcp-project-id  # BigQuery 사용 시
```

### 3. 파이프라인 실행 (최초 1회)

```bash
# 데이터 전처리 → 임베딩 → Pinecone 업로드
python src/pipeline.py --stage 5
```

### 4. 웹 앱 실행

```bash
streamlit run app.py
```
*최초 실행 시 Reranker 모델 다운로드로 인해 약 10~20초 소요될 수 있습니다.*

---

## 📁 프로젝트 구조

```
SKN22-3rd-2Team/
├── app.py                   # 🎯 Streamlit 웹 앱 (메인)
├── src/
│   ├── analysis_logic.py    # 분석 오케스트레이션 (검색+분석+스트리밍)
│   ├── patent_agent.py      # Self-RAG 에이전트 (Multi-Query + Claim Analysis)
│   ├── vector_db.py         # Pinecone + BM25 하이브리드 검색 (IPC 필터)
│   ├── reranker.py          # Cross-Encoder Reranker
│   ├── feedback_logger.py   # 피드백 수집 (JSONL)
│   ├── history_manager.py   # 분석 이력 관리 (SQLite)
│   ├── session_manager.py   # 세션 관리
│   ├── ui/                  # UI 컴포넌트
│   │   ├── components.py    # 결과 렌더링, 사이드바
│   │   ├── visualization.py # 특허 지형도 시각화
│   │   └── styles.py        # CSS 스타일
│   ├── preprocessor.py      # 4-Level 청구항 파서
│   ├── embedder.py          # OpenAI 임베딩
│   └── pipeline.py          # 파이프라인 오케스트레이터
├── logs/                    # 피드백 로그 및 시스템 로그
├── tests/                   # 🧪 DeepEval 및 단위 테스트
├── report/                  # 📄 기술 문서
├── requirements.txt
└── README.md
```

---

## 🔧 설정 옵션

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `OPENAI_API_KEY` | - | OpenAI API 키 (필수) |
| `PINECONE_API_KEY` | - | Pinecone API 키 (필수) |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | 임베딩 모델 |
| `GRADING_MODEL` | `gpt-4o-mini` | 관련성 평가 모델 |
| `ANALYSIS_MODEL` | `gpt-4o` | 최종 분석 모델 |
| `GRADING_THRESHOLD` | `0.6` | 재검색 기준 점수 |
| `TOP_K_RESULTS` | `5` | 검색 결과 개수 |

---

## 📊 분석 파이프라인 (Advanced)

```
[사용자 아이디어] & [IPC 필터]
        ↓
[Multi-Query Gen] 3가지 관점 쿼리 생성
        ↓
[Parallel Search] (Pinecone Dense + BM25 Sparse) x 3
        ↓
[IPC Filtering] 기술 분야 필터링
        ↓
[RRF Fusion] 검색 결과 통합 및 중복 제거
        ↓
[Reranker] Cross-Encoder 정밀 재정렬 (Top-5 선정)
        ↓
[Claim Analysis] 'All Elements Rule' 기반 청구항 정밀 분석
        ↓
[Streaming Output] 실시간 리포트 생성
```

---

## 🧪 테스트 및 QA

### 테스트 실행

```bash
# 모든 테스트 실행
pytest tests/ -v --asyncio-mode=auto

# RAG 품질 평가 (DeepEval)
pytest tests/test_evaluation.py -v
```

### 🏆 QA 현황 (100% Pass)

| 카테고리 | 테스트 항목 | 상태 | 비고 |
|---|---|---|---|
| **RAG Quality** | Faithfulness, Answer Relevancy | ✅ PASS | DeepEval 검증 |
| **Search Engine** | Hybrid RRF Logic | ✅ PASS | |
| **Parser** | 4-Level Claim Parsing | ✅ PASS | |
| **Data** | Integrity Check | ✅ PASS | |

> 상세 내용은 [03_test_report/README.md](03_test_report/README.md) 참조

---

## 💰 비용 정보

| 작업 | 예상 비용 |
|------|----------|
| BigQuery 쿼리 (10K 특허) | ~$2 (1회) |
| OpenAI 분석 (1건) | ~$0.01-0.03 |
| Pinecone 저장 | Serverless (사용량 기반) |

---

## 📄 라이선스

MIT License

---

## 👥 Team 뀨💕

**쇼특허** **(Short-Cut)** - AI 기반 특허 선행 기술 조사 시스템
