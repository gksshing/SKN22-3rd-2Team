# ⚡ 쇼특허 (Short-Cut)

**AI 기반 특허 선행 기술 조사 시스템**

사용자의 아이디어를 입력하면 기존 특허와 비교하여 **유사도**, **침해 리스크**, **회피 전략**을 분석해주는 Self-RAG 기반 특허 분석 도구입니다.

> **Team 뀨💕** | [기술 제안서](report/v3_technical_proposal.md) | [기술 리포트](report/v3_technical_report.md)

---

## 🎯 주요 기능

| 기능 | 설명 |
|------|------|
| **HyDE** | 사용자 아이디어를 가상 특허 청구항으로 변환하여 검색 품질 향상 |
| **Hybrid Search** | Pinecone (Dense) + Local BM25 (Sparse) + RRF 융합 검색 |
| **Serverless DB** | Pinecone 벡터 DB를 활용한 확장성 있는 데이터 관리 |
| **LLM Streaming** | 실시간 분석 결과 출력 (0초 체감 대기시간) |
| **4-Level Parser** | 다국어 청구항 파싱 (US/EP/KR 형식 지원) |
| **Grading Loop** | 검색 결과 관련성 평가, 자동 재검색 |
| **Critical CoT** | 유사도/침해/회피 분석 + 근거 특허 명시 |
| **QA Automation** | DeepEval 기반 RAG 품질 검증 (Faithfulness/Relevancy) |

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
conda create -n patent-guard python=3.11 -y
conda activate patent-guard

# 의존성 설치
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

---

## 📁 프로젝트 구조

```
SKN22-3rd-2Team/
├── app.py                   # 🎯 Streamlit 웹 앱 (루트 위치)
├── src/
│   ├── patent_agent.py      # Self-RAG 에이전트 (HyDE + Grading + CoT)
│   ├── vector_db.py         # Pinecone + BM25 하이브리드 검색
│   ├── preprocessor.py      # 4-Level 청구항 파서
│   ├── embedder.py          # OpenAI 임베딩
│   ├── pipeline.py          # 파이프라인 오케스트레이터
│   ├── config.py            # 설정 관리
│   └── data/
│       ├── raw/             # 원본 특허 데이터
│       ├── processed/       # 전처리된 데이터
│       ├── embeddings/      # 임베딩 벡터
│       └── index/           # 로컬 BM25 인덱스
├── tests/
│   ├── test_evaluation.py     # 🧪 DeepEval RAG 품질 테스트
│   ├── test_hybrid_search.py  # RRF 알고리즘 테스트
│   ├── test_parser.py         # 청구항 파서 테스트
│   └── conftest.py            # pytest 설정
├── report/
│   ├── v3_technical_proposal.md  # 기술 제안서
│   ├── v3_technical_report.md    # 기술 리포트
│   └── test_report*.html/txt     # 테스트 리포트
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

## 📊 분석 파이프라인

```
[사용자 아이디어]
        ↓
[HyDE] 가상 청구항 생성
        ↓
[Hybrid Search] Pinecone (Dense) + BM25 (Sparse)
        ↓
[RRF Fusion] 검색 결과 융합 (k=60)
        ↓
[Grading] 관련성 평가 (필요시 재검색)
        ↓
[Streaming Analysis] 실시간 상세 분석
        ↓
[분석 결과]
├── 유사도 평가 (0-100점)
├── 침해 리스크 (high/medium/low)
├── 구성요소 대비표
└── 회피 전략
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
