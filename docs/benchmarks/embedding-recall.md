# 임베딩 Recall@K 벤치마크

## 목적과 범위

[제공 계획서](../references/legal_embedding_recall_benchmark.md)를 기준으로 EVE-Bench-Legal-2.0에서 한국어 법률 판례 검색용 임베딩 모델을 비교합니다. 1차는 판례 문서 단위 Recall@K만 평가합니다. 2026-10-20 전체 평가를 완료했습니다. [결과와 모델 선택 근거](results/2026-10-20-embedding-recall.md)를 참조하세요.

| 항목 | 설계 |
| --- | --- |
| 모델 | `BAAI/bge-m3`, `nlpai-lab/KURE-v1`, `intfloat/multilingual-e5-large` |
| 프레임워크 | MTEB 2.26.11 지표 함수와 자체 Recall 교차 검증 |
| 검색 대상 | Qrels 후보만이 아닌 전체 Corpus |
| 입력 단위 | 판례 1개 = 벡터 1개 |
| 검색 방식 | Exact Top 100 |
| K | 1, 5, 10, 20, 50, 100 |
| 관련 문서 | Qrels relevance > 0 |
| 최종 집계 | 평가 가능한 질의별 Recall의 산술평균 |

공식 모델 카드와 접근 가능한 리비전을 확인하고 `configs/legal_recall.yaml`에 고정했습니다. 실제 모델 동작 확인은 probe, 전체 Recall 비교는 run으로 구분합니다.

## 실행 흐름

1. 같은 리비전·분할의 Corpus, Queries, Qrels를 로드하고 ID 정합성을 검증합니다.
2. 모델별 공식 규칙에 맞게 문서와 질의 입력을 구성합니다.
3. 문서·질의 벡터를 생성하거나 동일 조건의 캐시를 사용합니다.
4. 각 질의에 대해 전체 Corpus에서 정확 검색으로 상위 100개 문서를 구합니다.
5. 같은 검색 결과에서 모든 K의 Recall을 계산합니다.
6. 모델별 원본 JSON, 통합 CSV, 질의별 상세 결과와 실행 정보를 저장합니다.

## Recall 정의와 예외

`Recall@K(q) = |TopK(q) ∩ Relevant(q)| / |Relevant(q)|`

Relevant는 Qrels에서 relevance > 0인 문서 ID 집합입니다. 관련 판례 5개 중 상위 10개에서 3개를 찾았다면 Recall@10은 0.6입니다.

관련 문서가 0개인 질의는 집계에서 제외하고 ID·사유·제외 건수를 보고합니다. 모든 질의가 제외되면 점수는 0으로 기록하지 않고 평가 불가로 처리합니다. Qrels의 문서·질의 ID가 누락되거나 중복 결과가 생기는 정합성 문제는 조용히 제외하지 않고 검증 오류로 처리합니다.

K가 Corpus 크기보다 크면 전체 Corpus만 검색하며 Recall 분모는 관련 문서 수를 유지합니다. 동점은 문서 ID 기준 등 결정적인 순서로 처리하고 규칙을 기록합니다.

## MTEB 연동

검증된 Qrels와 정확 검색 순위를 `evaluation.py`의 `evaluate_with_mteb()`에 전달합니다. MTEB RetrievalEvaluator와 같은 지표 함수를 직접 사용하므로 별도 Task 객체나 Corpus 재변환이 필요하지 않습니다.

MTEB 2.26.11에서 양수 relevance를 이진 정답으로 변환하고 정답 없는 질의를 제외한 동일 검색 결과를 평가합니다. 여섯 K의 Recall을 프로젝트 집계와 교차 검증합니다. 요청 지표가 기본 제공되지 않으면 동일 Top 100 결과에서 별도 집계합니다. MTEB 원본 결과와 프로젝트 집계의 차이는 결과에 명시합니다.

## 비교 통제와 재현성

- 모든 모델에 동일한 Corpus·Queries·Qrels·리비전·분할을 사용합니다.
- 각 모델의 공식 Query/Passage Prefix, Pooling, 토큰 제한과 정규화 규칙을 따릅니다.
- 기본 유사도는 L2 정규화 벡터의 cosine이며, 공식 권장 방식이 우선하는 경우 차이와 이유를 기록합니다.
- 모델별 최대 토큰 길이, 문서·질의 입력 잘림 건수와 비율을 기록합니다. 모델 토크나이저에 따른 차이를 결과 해석에 반영합니다.
- 모델/데이터 리비전, 라이브러리 버전, 장비, dtype, batch size, 설정, 입력 해시와 실행 시각을 기록합니다.
- Qrels가 전체 관련 문서를 포괄한다고 단정하지 않습니다. 미판정 문서는 계산상 정답 집합에 포함되지 않지만 비관련성이 입증된 문서로 해석하지 않습니다.

## 결과 계약

`results/<run_id>/raw/`에 모델별 원본 JSON, `per_query/`에 모델별 질의 상세 JSON, `recall_summary.csv`에 모델별 R@1/5/10/20/50/100을 저장하도록 구현합니다. `manifest.json`에는 전체 질의 수·평가 수·제외 수와 사유, 실행 조건을 남깁니다.

질의별 결과에는 `query_id`, `query_text`, `relevant_doc_ids`, `retrieved_doc_ids_top100`, 검색 점수, `recall_at_1/5/10/20/50/100`, `model_id`, `model_revision`, `dataset_revision`을 포함합니다. 세부 필드명은 구현 시 스키마로 확정합니다. 점수는 실제 실행한 뒤에만 기록합니다.

## 1차 완료 기준

- [x] 원본 스키마와 분할을 확인하고 Corpus·Queries·Qrels 정합성을 검증했다.
- [x] 세 모델의 공식 규칙을 확인하고 문서·질의 벡터를 생성했다.
- [x] 각 모델로 전체 Corpus에서 Exact Top 100 검색을 수행했다.
- [x] Recall@1/5/10/20/50/100 및 평가·제외 질의 수를 계산했다.
- [x] 원본 JSON, 통합 CSV, 질의 상세 결과와 실행 정보를 저장했다.
- [x] 모델별 점수, 입력 잘림 비율과 Qrels 한계를 비교 문서에 기록했다.

청킹, nDCG, Hybrid Search, Reranker, Elasticsearch 운영 성능과 HNSW/ANN 평가는 후속 실험으로 분리합니다.
