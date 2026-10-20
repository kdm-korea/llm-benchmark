# 법률 도메인 Embedding 모델 Recall@K 벤치마크 계획

## 1. 목적

**EVE-Bench-Legal-2.0** 데이터셋을 활용해 한국어 법률 판례 검색에 적합한 Embedding 모델을 선정한다. 1차 실험에서는 **Recall@K**만 평가한다. 검색엔진(Elasticsearch), HNSW 근사 검색, Hybrid Search 및 Reranker는 평가 범위에서 제외한다.

## 2. 평가 범위 및 기본 설정

| 항목 | 설정 |
|---|---|
| 도메인 | 한국어 법률·판례 검색 |
| 데이터셋 | [EVE-Bench-Legal-2.0](https://huggingface.co/datasets/ysmeta/EVE-Bench-Legal-2.0) |
| 평가 프레임워크 | [MTEB](https://github.com/embeddings-benchmark/mteb) |
| 검색 방식 | 전체 Corpus 대상 Exact Vector Search |
| 유사도 | L2 정규화된 벡터의 Cosine Similarity (모델 권장 방식 우선) |
| 평가 지표 | Recall@1, Recall@5, Recall@10, Recall@20, Recall@50, Recall@100 |
| 정답 데이터 | 데이터셋 Qrels; relevance > 0인 문서를 관련 문서로 간주 |
| 결과물 | 모델별 원본 JSON, 통합 CSV, Query별 상세 결과 |

> 데이터셋의 실제 스키마, 파일명, 관련성 등급 및 평가 분할은 구현 전에 원본 리포지토리에서 확인한다. Qrels에 없는 문서를 반드시 비관련 문서로 단정할 수 없는 후보 풀 기반 판정의 한계도 기록한다.

## 3. 1차 비교 모델

| 모델 | Hugging Face ID | 비고 |
|---|---|---|
| BGE-M3 | `BAAI/bge-m3` | 다국어 검색 모델 |
| KURE-v1 | `nlpai-lab/KURE-v1` | 한국어 특화 후보 |
| Multilingual E5 Large | `intfloat/multilingual-e5-large` | Query/Passage 입력 Prefix 준수 |

모델마다 **Corpus와 Query를 동일한 모델의 규칙에 맞게** 임베딩한다. 입력 Prefix, 최대 토큰 길이, Pooling, 정규화 등 공식 모델 사용법을 따른다.

## 4. 전체 처리 흐름

```text
EVE-Bench-Legal-2.0
  ├── Corpus (판례 문서)
  ├── Queries (검색 질의)
  └── Qrels (질의별 관련 판례)
          │
          ▼
   Dataset Adapter
   (MTEB Custom Retrieval Task)
          │
          ▼
   Embedding Model별 실행
   ├── Corpus Embedding
   └── Query Embedding
          │
          ▼
   Exact Similarity Search
   (각 Query의 Top 100 검색)
          │
          ▼
   Qrels와 결과 비교
          │
          ▼
   Recall@1/5/10/20/50/100
          │
          ▼
   JSON / CSV / Query별 결과
```

### 단계별 작업

1. **데이터 로드:** Corpus, Query, Qrels를 읽고 문서 ID와 Query ID를 일관되게 매핑한다.
2. **입력 구성:** 판례 텍스트의 전처리 및 길이 제한을 적용한다. 1차 기준은 `판례 1개 = 벡터 1개`로 한다.
3. **모델별 임베딩:** Corpus와 Query를 각각 임베딩하고 벡터를 캐시해 재실행 비용을 줄인다.
4. **Exact 검색:** 각 Query와 전체 Corpus 간 유사도를 계산하고 상위 100개 문서를 추출한다. ANN을 사용하지 않는다.
5. **정답 비교:** Qrels에서 relevance > 0인 문서 집합을 만들고 Recall@K를 계산한다.
6. **집계:** 평가 가능한 Query별 Recall@K를 산술 평균해 모델별 점수를 구한다.
7. **결과 저장:** 모델별 점수, Query별 검색 결과, 환경 및 버전 정보를 저장한다.

## 5. Recall@K 정의

질의 `q`에 대해:

\[
\operatorname{Recall@K}(q) =
\frac{|\operatorname{TopK}(q) \cap \operatorname{Relevant}(q)|}
{|\operatorname{Relevant}(q)|}
\]

- `TopK(q)`: 임베딩 유사도 기준 상위 K개 검색 문서 ID
- `Relevant(q)`: Qrels에서 관련성 등급이 1 이상인 문서 ID
- 관련 문서가 0개인 Query는 집계에서 제외하고 제외 건수를 별도 보고한다.
- 최종 점수는 **Query별 Recall@K의 평균(Macro Average)**으로 계산한다.

**예시:** 관련 판례가 5개이고 Top 10에서 그중 3개를 찾으면 `Recall@10 = 3/5 = 0.6`이다.

## 6. MTEB 적용 방식

EVE-Bench-Legal-2.0을 MTEB가 바로 인식하지 못할 경우 **Custom Retrieval Task Adapter**를 구현한다.

- `corpus`: 문서 ID, 제목, 본문
- `queries`: Query ID, 질의 텍스트
- `relevant_docs`: Query ID → `{문서 ID: 관련성 등급}`
- MTEB Retrieval Evaluator로 검색 품질을 평가한다.
- 설치된 MTEB 버전에서 지원하는 Recall K값과 검색 Top K 한도를 확인한다. 기본 설정에서 제공하지 않는 K는 검색 결과를 기반으로 추가 집계한다.
- 모델별 인코딩 인터페이스가 다른 경우 MTEB 모델 Adapter를 구성한다.

> MTEB 버전에 따라 Custom Task API와 결과 저장 인터페이스가 달라질 수 있으므로 구현 시 라이브러리 버전을 고정한다.

## 7. 프로젝트 디렉터리

```text
legal-embedding-benchmark/
├── configs/
│   └── models.yaml
├── datasets/
│   └── eve_legal/
│       ├── corpus.jsonl
│       └── qrels.jsonl
├── src/
│   ├── tasks/
│   │   └── eve_legal.py
│   ├── benchmark.py
│   ├── metrics.py
│   └── aggregate.py
├── cache/
│   └── embeddings/
├── results/
│   ├── raw/
│   ├── per_query/
│   └── recall_summary.csv
├── requirements.txt
└── README.md
```

파일명과 필드는 데이터셋 원본 구조 확인 후 조정한다.

## 8. 결과 파일 설계

### `results/recall_summary.csv`

| Model | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 |
|---|---:|---:|---:|---:|---:|---:|
| BGE-M3 | - | - | - | - | - | - |
| KURE-v1 | - | - | - | - | - | - |
| Multilingual E5 Large | - | - | - | - | - | - |

점수는 **실제 실행 후 기록**한다. 현재 표는 결과 형식 예시다.

### Query별 상세 결과

각 Query마다 다음 항목을 저장한다.

- `query_id`, `query_text`
- `relevant_doc_ids`, `retrieved_doc_ids_top100`
- `recall_at_1`, `recall_at_5`, `recall_at_10`, `recall_at_20`, `recall_at_50`, `recall_at_100`
- `model_id`, `model_revision`, `dataset_revision`

## 9. 공정한 비교를 위한 통제 조건

1. **동일한 Corpus / Query / Qrels:** 모든 모델에 같은 평가 데이터를 사용한다.
2. **전체 문서 검색:** Qrels의 정답 후보만 대상으로 검색하지 않는다.
3. **Exact 검색 고정:** ANN 탐색 오차를 제거해 모델 간 비교를 우선한다.
4. **모델별 공식 인코딩 규칙:** Query/Passage Prefix 및 Pooling 방식 등을 준수한다.
5. **긴 판례 처리:** 문서가 최대 입력 토큰을 초과해 잘리는 비율을 측정한다. 1차는 문서당 1벡터, Chunking은 후속 실험으로 분리한다.
6. **재현성:** 데이터셋/모델 Revision, 라이브러리 버전, 실행 환경, Batch Size를 기록한다.
7. **평가 편향:** Qrels가 불완전할 수 있다는 점을 결과 해석에 반영한다.

## 10. 1차 완료 기준

- [ ] 데이터셋을 로드하고 Corpus / Query / Qrels 정합성을 검증했다.
- [ ] 세 Embedding 모델의 Corpus 및 Query 벡터를 생성했다.
- [ ] 각 모델로 전체 Corpus에 대해 Exact Top 100 검색을 수행했다.
- [ ] Recall@1/5/10/20/50/100을 계산했다.
- [ ] 통합 CSV 및 Query별 결과 JSON을 저장했다.
- [ ] 모델별 Recall@K를 비교하고 입력 잘림 비율을 기록했다.

## 11. 후속 단계 (이번 범위 제외)

1. **NDCG@10 및 순위 품질 분석:** 관련성 등급과 검색 순위를 반영한다.
2. **Chunking 실험:** 문서 단위 임베딩과 청크 단위 임베딩을 비교한다.
3. **Hybrid Search:** BM25 + Dense 검색 결과를 RRF 또는 가중 합산으로 결합한다.
4. **Elasticsearch 운영 평가:** HNSW Recall, `num_candidates`, p95 Latency, 리소스 사용량을 측정한다.

## 참고 링크

- [EVE-Bench-Legal-2.0](https://huggingface.co/datasets/ysmeta/EVE-Bench-Legal-2.0)
- [MTEB GitHub](https://github.com/embeddings-benchmark/mteb)
- [Sentence Transformers](https://www.sbert.net/)
