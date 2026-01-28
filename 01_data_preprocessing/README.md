# 📊 데이터 수집 및 전처리 보고서

> **⚡ 쇼특허 (Short-Cut) v3.0 - AI 특허 선행 기술 조사 시스템**  
> Team: 뀨💕 | 작성일: 2026-01-28

---

## 1. 데이터 수집 개요

### 1.1 데이터 소스

| 항목 | 내용 |
|------|------|
| **소스** | Google Patents Public Dataset |
| **저장소** | Google BigQuery (`patents-public-data.patents.publications`) |
| **접근 방식** | BigQuery SQL 쿼리 |
| **비용** | ~$2 USD (390GB 스캔) |

### 1.2 수집 기준

| 항목 | 설정값 |
|------|--------|
| **기간** | 2018-01-01 ~ 2024-12-31 |
| **국가** | US, EP, WO, CN, JP, KR |
| **수집량** | 10,000건 |

### 1.3 도메인 키워드

```
AI/NLP 도메인 키워드:
- retrieval augmented generation
- large language model
- neural information retrieval
- semantic search
- document embedding
- transformer attention
- knowledge graph reasoning
- prompt engineering
- context window
- fine-tuning language model
- quantization neural network
- efficient inference
- multi-modal retrieval
```

### 1.4 IPC 분류 코드

| IPC 코드 | 분류 |
|----------|------|
| G06F 16 | Information Retrieval |
| G06F 40 | Natural Language Processing |
| G06N 3 | Neural Networks |
| G06N 5 | Knowledge-based Systems |
| G06N 20 | Machine Learning |
| H04L 12 | Data Switching Networks |

---

## 2. 수집된 데이터 현황

### 2.1 원본 데이터 (Raw Data)

| 파일명 | 크기 | 건수 |
|--------|------|------|
| `patents_10k.json` | 74 MB | 10,000건 |

### 2.2 전처리 데이터 (Processed Data)

| 파일명 | 크기 | 건수 |
|--------|------|------|
| `processed_patents_10k.json` | 61 MB | 10,000건 |

### 2.3 임베딩 데이터 (v3.0 신규)

| 파일명 | 크기 | 벡터 수 |
|--------|------|---------|
| `embeddings_processed_patents_10k.npz` | ~120 MB | 20,664개 |

### 2.4 인덱스 파일 (v3.0 신규)

| 파일명 | 용도 | 타입 |
|--------|------|------|
| `patent_index.bin` | FAISS Dense 검색 (Deprecated) | IndexFlatIP |
| `pinecone_index` | Pinecone Serverless | Cosine Metric |
| `bm25_index.pkl` | BM25 Sparse 검색 | rank-bm25 |
| `chunk_metadata.pkl` | 청크 메타데이터 | Python dict |

### 2.5 데이터 필드 구조

```json
{
  "publication_number": "US-12345678-A1",
  "title": "특허 제목",
  "abstract": "특허 초록 텍스트...",
  "claims": [
    {
      "claim_number": 1,
      "claim_type": "independent",
      "claim_text": "청구항 텍스트...",
      "parent_claim": null,
      "rag_components": ["retrieval", "embedding"]
    }
  ],
  "ipc_codes": ["G06N 3/08", "G06F 40/30"],
  "cited_publications": ["US-98765432-B2"],
  "filing_date": "2023-01-15",
  "assignee": "기업명"
}
```

---

## 3. 전처리 과정 (v3.0 향상)

### 3.1 전처리 파이프라인

```
[원본 데이터]
     ↓
[Stage 1] BigQuery 추출
     ↓
[Stage 2] 텍스트 정규화
     - 특수문자 처리
     - 공백 정리
     - 인코딩 통일 (UTF-8)
     ↓
[Stage 3] 4-Level 청구항 파싱 (v3.0 신규)
     - Level 1: Regex (US/EP/KR 패턴)
     - Level 2: 구조 분석 (들여쓰기/번호)
     - Level 3: NLP 문장 경계 (Spacy)
     - Level 4: 최소 분할 (Fallback)
     ↓
[Stage 4] 청킹 (Chunking)
     - 최대 1024 토큰 단위 분할
     - 오버랩 128 토큰
     ↓
[Stage 5] 임베딩 생성
     - 모델: text-embedding-3-small
     - 차원: 1536
     ↓
[Stage 6] 인덱스 생성 (v3.0 신규)
     - Pinecone Serverless (Dense) - Upsert
     - BM25 인덱스 (Sparse) - Local Save
     ↓
[Hybrid Search 준비 완료]
```

### 3.2 4-Level 청구항 파서 상세 (v3.0 신규)

| Level | 방법 | 지원 포맷 | 우선순위 |
|-------|------|-----------|----------|
| 1 | Regex | `1.`, `Claim 1:`, `(1)`, `[1]`, `제1항:` | 높음 |
| 2 | 구조 분석 | 들여쓰기, 번호 체계 | 중간 |
| 3 | NLP (Spacy) | 문장 경계 탐지 | 중간 |
| 4 | 최소 분할 | 문단 분리 또는 전체 1개 | 낮음 (Fallback) |

**지원 포맷:**
- US/EP: `1. A method comprising...`
- 괄호: `(1) A method...`, `[1] A method...`
- 한국어: `제1항:`, `청구항 1:`
- 종속항: `The method of claim 1...`, `제1항에 있어서...`

### 3.3 전처리 통계

| 항목 | 수치 |
|------|------|
| 원본 특허 수 | 10,000건 |
| 전처리 완료 | 10,000건 |
| 추출된 청구항 | ~30,000개 |
| 생성된 청크 | ~20,664개 |
| 임베딩 벡터 | 20,664개 (1536차원) |
| Pinecone 인덱스 | 20,664 벡터 (Serverless) |
| BM25 문서 | 20,664개 |

### 3.4 품질 검증

| 검증 항목 | 결과 |
|-----------|------|
| NULL 값 비율 | < 5% |
| 영어 Abstract 보유율 | ~70% |
| 청구항 파싱 성공률 | **~95%** (v3.0 향상) |
| IPC 코드 보유율 | 100% |
| RAG 컴포넌트 탐지 | ~60% (도메인 키워드 기반) |

---

## 4. 하이브리드 검색 준비 (v3.0 신규)

### 4.1 검색 인덱스 구조

```
[사용자 쿼리]
     ↓
[HyDE] 가상 청구항 생성
     ↓
     ├──→ [Pinecone Dense] 벡터 유사도 검색
     │          ↓
     │    Top-K (semantic)
     │
     └──→ [BM25 Sparse] 키워드 매칭
                ↓
          Top-K (lexical)
     ↓
[RRF Fusion] k=60, weight=0.5:0.5
     ↓
[최종 Top-K 결과]
```

### 4.2 RRF 설정

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `rrf_k` | 60 | RRF 상수 (높을수록 상위 랭크 영향 완화) |
| `dense_weight` | 0.5 | Dense 검색 가중치 |
| `sparse_weight` | 0.5 | Sparse 검색 가중치 |

---

## 5. 데이터 활용 계획

### 5.1 Self-RAG 분석 흐름

```
사용자 아이디어 입력
     ↓
HyDE (가상 청구항 생성)
     ↓
Hybrid Search (FAISS + BM25 + RRF)
     ↓
관련성 평가 (Grading)
     ↓
Streaming 분석 (유사도/침해/회피)
```

### 5.2 데이터 제한 사항

| 항목 | 내용 |
|------|------|
| **샘플 크기** | 10,000건 (전체 특허의 <0.01%) |
| **용도** | 데모/프로토타입용 |
| **제한** | 종합적 선행 기술 조사에는 부적합 |

---

## 6. 파일 위치

```
SKN22-3rd-2Team/
├── app.py                          # Streamlit 웹 앱
├── src/
│   ├── preprocessor.py             # 4-Level 청구항 파서
│   ├── embedder.py                 # OpenAI 임베딩
│   ├── vector_db.py                # FAISS + BM25 하이브리드
│   ├── pipeline.py                 # 파이프라인 실행
│   └── data/
│       ├── raw/
│       │   └── patents_10k.json    # 원본 데이터
│       ├── processed/
│       │   └── processed_patents_10k.json
│       ├── embeddings/
│       │   └── embeddings_*.npz    # 임베딩 벡터
│       └── index/
│           ├── patent_index.bin    # FAISS 인덱스
│           ├── bm25_index.pkl      # BM25 인덱스
│           └── chunk_metadata.pkl  # 메타데이터
├── tests/
│   ├── test_evaluation.py          # 🧪 DeepEval RAG 품질 테스트
│   ├── test_hybrid_search.py       # RRF 테스트
│   └── test_parser.py              # 파서 테스트
└── 01_data_preprocessing/
    └── README.md                   # 본 보고서
```

---

## 7. 실행 방법

### 전체 파이프라인 실행

```bash
# Stage 1-5: 데이터 추출 → 전처리 → 임베딩 → 인덱싱
python src/pipeline.py --stage 5
```

### 단계별 실행

```bash
# Stage 1: BigQuery 추출
python src/pipeline.py --stage 1 --limit 10000

# Stage 2: 전처리
python src/pipeline.py --stage 2

# Stage 3: 청킹
python src/pipeline.py --stage 3

# Stage 4: 임베딩
python src/pipeline.py --stage 4

# Stage 5: 인덱싱 (FAISS + BM25)
python src/pipeline.py --stage 5
```

### 테스트 실행

```bash
# 파서 테스트
pytest tests/test_parser.py -v

# 하이브리드 검색 테스트
pytest tests/test_hybrid_search.py -v
```

---

## 8. 참고 자료

- [Google Patents Public Dataset](https://console.cloud.google.com/marketplace/product/google_patents_public_datasets/google-patents-public-data)
- [BigQuery 가격 정책](https://cloud.google.com/bigquery/pricing)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [rank-bm25](https://github.com/dorianbrown/rank_bm25)

---

*작성: ⚡ 쇼특허 (Short-Cut) Team - 뀨💕*
