# 개발 아키텍처

법률 RAG 연구를 위해 임베딩 검색 실험과 평가 데이터 구축을 관리합니다. 상위 패키지는 `llm_benchmark`이며, 공통 유틸리티와 임베딩 실험 코드를 구분합니다.

## 코드 구조

```text
src/llm_benchmark/
  common/
    hashing.py       # 파일 SHA-256·기존 JSON 해시 형식 유지
    io.py            # 원자적 JSON·JSONL 읽기/쓰기
  embedding/
    dataset.py       # EVE 다운로드, 데이터 검증, 정규화, JSON 저장
    embeddings.py    # 모델 로드, 입력 구성, 임베딩, 캐시 검증
    evaluation.py    # 전체 Corpus 정확 검색, Recall 계산, MTEB 검증
    run.py           # 임베딩 CLI, 실험 순서, probe, 결과 저장
```

`embedding/run.py`의 `run_benchmark()`부터 읽으면 임베딩 실험 전체를 따라갈 수 있습니다. 기존 `embedding-benchmark` 명령은 이 모듈을 실행합니다.

의존 방향은 `embedding → common`입니다. `common`은 표준 라이브러리만 사용하고 실험 코드를 import하지 않습니다. 법률 데이터 규칙·검색 지표·모델 설정은 임베딩 실험에 둡니다. 파일 저장과 해시 함수는 공통으로 분리하되 기존 질의 ID·캐시 키 형식은 유지합니다.

임베딩 의존성은 `embedding` extra로 설치합니다. 테스트 fixture와 MTEB 캐시는 해당 실험의 임시 폴더를 사용하며 환경변수는 테스트 후 복구합니다. `--import-mode=importlib`로 테스트 파일 이름 충돌을 방지합니다. `tests/architecture/`는 실험 폴더를 자동 발견하고 교차 import 및 공통 모듈의 실험 의존을 검사합니다.

1. `prepare_dataset()`으로 데이터를 검증합니다.
2. `_encode_dataset()`으로 문서와 질의 벡터를 준비합니다.
3. `exact_search()`로 전체 Corpus의 Top 100을 구합니다.
4. `evaluate_recall()`과 `evaluate_with_mteb()`로 같은 검색 결과를 평가합니다.
5. 완료한 모델의 JSON·CSV와 실행 manifest를 저장합니다.

데이터만 확인할 때는 `dataset.py`, 모델 입력과 캐시는 `embeddings.py`, 지표는 `evaluation.py`를 보면 됩니다. 모델 메모리는 검색 전에 해제하며, 실패한 실행도 manifest를 남깁니다.

## 기술과 평가 조건

- Python과 `pyproject.toml` / uv를 사용하고 의존성은 `uv.lock`에 고정합니다.
- `configs/legal_recall.yaml`에서 모델·데이터 리비전과 실행 조건을 관리합니다.
- 판례 1개당 벡터 1개를 생성하고 전체 Corpus에서 cosine 정확 검색을 수행합니다.
- 문서·질의 ID, 원본 해시, 캐시 키와 결과 파일 형식은 구조 변경 전과 동일하게 유지했습니다.
- MTEB 2.26.11의 RetrievalEvaluator가 사용하는 지표 함수를 직접 호출합니다. 이미 순위가 있으므로 별도 Custom Task 생성과 데이터 변환을 거치지 않습니다.
- MTEB의 반올림된 Recall과 자체 계산을 비교하고, 차이가 허용 범위를 벗어나면 실패합니다.

`core`, `application`, `adapters`, `domains`, 별도 CLI 패키지와 `BenchmarkBackend`는 제거했습니다. 법률 데이터 규칙은 `dataset.py`에 모았습니다. 다른 도메인이나 구현체가 실제로 필요해지면 그때 분리를 검토합니다.

## 데이터와 문서

```text
configs/                   실험 설정
data/raw/                 원본 데이터
data/prepared/            정규화 데이터와 검증 보고서
cache/                    모델·임베딩 캐시
results/<run_id>/         원본 JSON, 질의별 결과, CSV, manifest
results/probes/           소량 실제 추론 확인 결과
tests/embedding/          기존 임베딩 실험 회귀 테스트
tests/common/             공통 저장·해시 호환성 검증
tests/architecture/       실험 간 의존 경계 검증

docs/engineering/         개발 기술과 실행 방법
docs/domain/legal/        법률 데이터와 정답 규칙
docs/benchmarks/          평가 설계
docs/prompts/             시스템 프롬프트
docs/references/          제공 원본 문서
docs/work-log/            실제 수행 기록
```

자세한 명령은 [실행 방법](running.md), 입력 규칙은 [모델 설정](models.md), 법률 데이터 규칙은 [데이터셋 문서](../domain/legal/dataset.md)를 참조합니다.
