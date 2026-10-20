# embedding-benchmark

한국어 법률·판례 검색에 사용할 임베딩 모델을 비교하고 모델 선택의 근거를 남기는 프로젝트입니다.

[제공된 벤치마크 계획서](docs/references/legal_embedding_recall_benchmark.md)를 바탕으로 EVE-Bench-Legal-2.0의 전체 Corpus에서 정확 검색을 수행하고, 판례 문서 단위 Recall@K를 평가하도록 설계했습니다. 데이터 다운로드·정합성 검증과 실행 파이프라인을 구현했습니다. 세 모델의 실제 문서·질의 추론 확인과 테스트 17개가 통과했습니다. 전체 Recall 평가는 아직 실행하지 않았습니다.

## 1차 실험

| 항목 | 계획 |
| --- | --- |
| 데이터셋 | EVE-Bench-Legal-2.0 |
| 모델 | BAAI/bge-m3, nlpai-lab/KURE-v1, intfloat/multilingual-e5-large |
| 평가 프레임워크 | MTEB 2.26.11 지표 교차 검증 |
| 입력 단위 | 판례 1개 = 벡터 1개 |
| 검색 | 전체 Corpus 대상 Exact Top 100 |
| 지표 | Recall@1/5/10/20/50/100의 질의별 산술평균 |
| 정답 | Qrels relevance > 0 |
| 결과 | 모델별 원본 JSON, 통합 CSV, 질의별 상세 결과 |

데이터셋과 세 모델의 리비전을 고정하고 MTEB 2.26.11을 연결했습니다. Corpus 59,786개와 고유 질의 5,998개를 검증했습니다. 비교 점수는 전체 평가를 실제 실행한 뒤 기록합니다.

## 코드 구조

```text
src/embedding_benchmark/
  dataset.py       # 법률 데이터 로드·검증
  embeddings.py    # 모델 실행·벡터 캐시
  evaluation.py    # 정확 검색·Recall·MTEB
  run.py           # CLI·실험 흐름·결과 저장
```

실험 흐름은 `run.py`의 `run_benchmark()`부터 읽으면 됩니다. 개발 문서와 법률 도메인 문서는 분리해 관리합니다.

## 문서 안내

- [문서 목차](docs/README.md)
- [개발 아키텍처](docs/engineering/architecture.md)
- [설치 및 실행 방법](docs/engineering/running.md)
- [실제 데이터 확인 기록](docs/domain/legal/data-inspection-2026-10-19.md)
- [Recall@K 평가 설계와 완료 기준](docs/benchmarks/embedding-recall.md)
- [법률 데이터셋 및 정답 관리](docs/domain/legal/dataset.md)
- [시스템 프롬프트 관리](docs/prompts/README.md)
- [수행 작업 기록](docs/work-log/README.md)

프로젝트명은 `embedding-benchmark`, Python 패키지명은 `embedding_benchmark`입니다. 현재 작업 폴더명 `llm-benchmark`는 유지합니다. 실행 명령은 아래와 같습니다. `probe`는 실제 추론 확인이며, `run`이 전체 Corpus를 검색하는 Recall 평가입니다.

```bash
.venv/bin/embedding-benchmark prepare
.venv/bin/embedding-benchmark probe
.venv/bin/embedding-benchmark run
.venv/bin/python -m pytest -q
```
