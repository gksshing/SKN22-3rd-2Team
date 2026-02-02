# ⚡ 쇼특허 (Short-Cut)

**AI 기반 특허 선행 기술 조사 시스템**

사용자의 아이디어를 입력하면 기존 특허와 비교하여 **유사도**, **침해 리스크**, **회피 전략**을 분석해주는 RAG 기반 특허 분석 도구입니다.


---

## 🎯 주요 기능

| 기능 | 설명 |
|------|------|
| **HyDE** | 사용자 아이디어로부터 '가상 청구항'을 생성하여 검색 재현율(Recall)을 획기적으로 향상 |
| **Multi-Query RAG** | 기술적/법적/문제해결 관점으로 쿼리를 자동 확장하여 검색 누락 최소화 |
| **Hybrid Search** | Pinecone Serverless (Dense + Sparse) 통합 인덱스 검색으로 정확도 극대화 |
| **RAG Evaluation** | **DeepEval** 프레임워크를 활용한 답변의 충실도(Faithfulness) 및 관련성(Relevancy) 자동 검증 |
| **Reranker** | Cross-Encoder 모델을 활용하여 검색 결과의 관련성 정밀 재정렬 |
| **Intelligent Parsing** | Regex → 구조 분석 → NLP → Fallback의 **4-Level 청구항 파싱**으로 비정형 데이터 정복 |
| **Claim-Level Analysis** | '모든 구성요소 법칙(All Elements Rule)'을 적용한 특허 침해 리스크 정밀 진단 |
| **Feedback Loop** | 분석 결과에 대한 사용자 피드백(👍/👎) 수집 및 검색 품질 개선 엔진 |
| **Guardian Map** | 아이디어를 성(Castle)으로, 위협 특허를 침입자로 시각화한 **직관적 방어 전략 지도** |
| **PDF Report** | 분석된 모든 내용을 깔끔한 PDF 보고서로 자동 생성 및 다운로드 |
| **지식재산권 용어 사전 제공** | 특허 분야가 생소한 사용자를 위해 필수 지식재산권 용어를 정리한 PDF 가이드를 사이드바에서 상시 제공 |

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
conda create -n shortcut python=3.11 -y
conda activate shortcut

# 의존성 설치
pip install -r requirements.txt

# NLP 모델 다운로드
python -m spacy download en_core_web_sm
```

### 2. 환경 변수 설정

`.env` 파일을 프로젝트 루트에 생성하고 아래 항목을 입력합니다:
```env
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
GCP_PROJECT_ID=your-gcp-project-id  # BigQuery 연동 시 필요
```

### 3. 데이터 인덱싱 (최초 1회)

BigQuery 추출부터 Pinecone 업로드까지 전체 파이프라인을 실행합니다:
```bash
# 전체 파이프라인 실행 (Extraction -> Preprocessing -> Embedding -> Indexing)
python -m src.pipeline --limit 10000 --execute
```
*(이미 처리된 데이터가 있다면 `python -m src.pipeline --stage 5`로 인덱싱만 수행 가능)*

### 4. 웹 앱 실행

```bash
streamlit run app.py
```

---

## 📊 분석 아키텍처

```mermaid
flowchart TD
    A[사용자 아이디어] --> H[HyDE: 가상 청구항 생성]
    H --> B[Multi-Query 확장]
    B --> C{하이브리드 검색}
    C -->|Dense + Sparse| D[Pinecone Serverless]
    D --> F[RRF Fusion & Reranking]
    F --> G[LLM: 청구항 정밀 분석]
    G --> I[실시간 스트리밍 Analysis]
    I --> J[Guardian Map 시각화]
    I --> K[PDF 리포트 생성]
```

---

## 📁 프로젝트 구조

```
SKN22-3rd-2Team/
├── app.py                   # Streamlit 메인 애플리케이션
├── src/
│   ├── analysis_logic.py    # 분석 프로세스 오케스트레이션
│   ├── patent_agent.py      # AI 에이전트 (HyDE + Multi-Query + Claim Analysis)
│   ├── vector_db.py         # 하이브리드 검색 엔진 (Pinecone + BM25)
│   ├── reranker.py          # 정밀 재정렬 모델
│   ├── ui/                  # UI 컴포넌트
│   │   ├── visualization.py # Guardian Map 시각화 로직
│   │   └── components.py    # 리포트 UI
│   ├── preprocessor.py      # 4-Level 지능형 청구항 파서
│   └── pdf_generator.py     # 결과 리포트 PDF 생성기
├── logs/                    # 시스템 및 피드백 로그
├── tests/                   # 품질 검증 테스트 (DeepEval)
└── report/                  # 기술 문서 및 분석 보고서
```

---

## 🔧 주요 설정 (src/config.py)

| 설정 항목 | 설명 |
|------|------|
| `PINECONE_INDEX_NAME` | Pinecone 서버리스 인덱스 이름 |
| `ANALYSIS_MODEL` | 분석에 사용하는 LLM 모델 (Default: gpt-4o) |
| `TOP_K_RESULTS` | 최종 분석에 반영할 특허 개수 |
| `HYBRID_ALPHA` | Dense/Sparse 검색 가중치 비율 |

---

## 📄 라이선스

MIT License

---

## 👥 Team 뀨💕
**쇼특허 (Short-Cut)** - "특허 분석의 지름길을 제시합니다." 
