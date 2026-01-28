# 🧪 테스트 계획 및 결과 보고서

> **⚡ 쇼특허 (Short-Cut) v3.0 - AI 특허 선행 기술 조사 시스템**  
> Team: 뀨💕 | 작성일: 2026-01-28  
> 테스트 프레임워크: pytest 9.0.2

---

## 1. 테스트 개요

### 1.1 테스트 범위

| 모듈 | 파일 | 테스트 수 | 커버리지 |
|------|------|----------|----------|
| **DeepEval RAG Quality** | `test_evaluation.py` | 4 | N/A |
| **Hybrid Search (RRF)** | `test_hybrid_search.py` | 8 | 100% |
| **Claim Parser (4-Level)** | `test_parser.py` | 19 | 100% |
| **Total** | - | **31** | **100% Pass** |

### 1.2 테스트 환경

| 항목 | 값 |
|------|-----|
| **OS** | Windows 11 (10.0.26100) |
| **Python** | 3.11.14 |
| **pytest** | 9.0.2 |
| **DeepEval** | 0.21.0 |
| **LLM Model** | gpt-4o-mini (Evaluation) |

---

## 2. 테스트 결과 요약

```
============================= test session starts =============================
platform win32 -- Python 3.11.14, pytest-9.0.2
collected 31 items

tests/test_evaluation.py ....                                            [ 12%]
tests/test_hybrid_search.py ........                                     [ 38%]
tests/test_parser.py ...................                                 [100%]

============================= 31 passed in 145.22s ============================
```

| 결과 | 수치 |
|------|------|
| ✅ **Passed** | 31 |
| ❌ Failed | 0 |
| **Pass Rate** | **100%** |

---

## 3. RAG 품질 검증 (DeepEval)

📄 **파일**: `tests/test_evaluation.py`

### 3.1 평가 메트릭

| 메트릭 | 설명 | Threshold |
|--------|------|-----------|
| **FaithfulnessMetric** | 답변이 검색된 특허(Context)에 근거하는지 검증 (Hallucination 방지) | 0.7 |
| **AnswerRelevancyMetric** | 답변이 사용자 질문(Query)과 관련 있는지 검증 | 0.7 |

### 3.2 테스트 시나리오 (AI/NLP 도메인)

| ID | 테스트명 | 쿼리 주제 | 결과 |
|----|----------|----------|------|
| `test_001` | **RAG 검색 시스템** | Retrieval, Embedding, Vector Search | ✅ PASS |
| `test_002` | **Semantic Search** | Transformer, Cosine Similarity, Neural IR | ✅ PASS |
| `test_003` | **LLM Fine-tuning** | Quantization, Prompt Engineering | ✅ PASS |
| `single` | **Quick Check** | 자연어 처리 기반 특허 검색 | ✅ PASS |

---

## 4. Hybrid Search (RRF) 테스트

📄 **파일**: `tests/test_hybrid_search.py`

### 4.1 테스트 시나리오

RRF (Reciprocal Rank Fusion) 알고리즘의 정확성을 검증합니다.

```
RRF_score(d) = Σ weight / (k + rank + 1)
```

### 4.2 주요 테스트 케이스

| # | 테스트명 | 설명 | 상태 |
|---|---------|------|------|
| 1 | `test_cross_rank_verification` | Dense/Sparse 상위 문서 랭킹 검증 | ✅ |
| 2 | `test_symmetric_weighting` | 0.5:0.5 가중치 균형 검증 | ✅ |
| 3 | `test_asymmetric_weighting` | 비대칭 가중치(0.8:0.2) 동작 검증 | ✅ |
| 4 | `test_edge_case_*` | 빈 결과, 단일 소스 결과 처리 | ✅ |

---

## 5. Claim Parser (4-Level) 테스트

📄 **파일**: `tests/test_parser.py`

### 5.1 테스트 전략

4-Level Fallback 파서의 각 레벨별 동작을 검증합니다.

```
Level 1: Regex Pattern → Level 2: Structure → Level 3: NLP → Level 4: Minimal
```

### 5.2 주요 테스트 케이스

| Level | 설명 | 상태 |
|-------|------|------|
| **Level 1** | US/EP 표준 형식("1. A method...") 및 번호 체계 파싱 | ✅ |
| **Level 2** | 괄호/대괄호("(1)", "[1]") 및 들여쓰기 구조 파싱 | ✅ |
| **Level 3** | OCR 노이즈("C1aim") 처리 및 문장 경계 탐지 | ✅ |
| **Level 4** | 구조 없는 텍스트의 문단 단위 폴백 처리 | ✅ |

---

## 6. 테스트 실행 방법

### 6.1 전체 테스트 실행

```bash
# 기본 실행
pytest tests/ -v --asyncio-mode=auto

# 상세 출력
pytest tests/ -v --tb=short
```

### 6.2 DeepEval RAG 품질 테스트

```bash
# RAG 품질 테스트만 실행 (OpenAI API 비용 발생)
pytest tests/test_evaluation.py -v
```

### 6.3 HTML 리포트 생성

```bash
# HTML 리포트 생성
pytest tests/ --html=report/test_report.html --self-contained-html
```

---

## 7. 향후 테스트 계획

| 우선순위 | 항목 | 예상 일정 |
|----------|------|----------|
| 🔴 High | OpenAI API Mock 서버 구축 (비용 절감) | 1주 |
| 🟡 Medium | FAISS 인덱스 I/O 통합 테스트 | 1주 |
| 🟢 Low | Streamlit E2E UI 테스트 | 2주 |

---

*작성: ⚡ 쇼특허 (Short-Cut) Team - 뀨💕*
