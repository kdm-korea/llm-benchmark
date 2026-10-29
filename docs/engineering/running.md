# 실행 방법

프로젝트 루트에서 실행합니다. Python 3.11 이상 3.15 미만을 지원하도록 설정했고, 이번 환경은 Python 3.12로 검증했습니다. 의존성 전체 버전은 `uv.lock`에 고정했습니다.

## 환경 설치

uv가 있는 환경:

```bash
uv sync --locked --extra embedding --cache-dir cache/uv
```

이번 로컬 환경에서는 uv가 없어 아래 순서로 가상환경에 설치했습니다. 다시 설치할 필요 없이 현재 `.venv/bin/embedding-benchmark`를 실행할 수 있습니다.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install uv
.venv/bin/uv sync --locked --extra embedding --cache-dir cache/uv
```

모델 및 데이터 다운로드에는 네트워크가 필요합니다. 데이터, 모델, 벡터와 MTEB 캐시는 프로젝트의 `data/` 및 `cache/`에서 관리합니다.

### 실험별 의존성·테스트 분리

기본 패키지와 `common`은 표준 라이브러리만 사용합니다. `embedding` extra는 임베딩 실험 의존성을, `koblex` extra는 PyArrow만 추가합니다. 같은 환경에 둘 다 필요하면 `--all-extras`를 사용합니다. 서로의 설치 상태까지 분리하려면 프로젝트 루트에서 별도 환경을 만듭니다.

```bash
UV_PROJECT_ENVIRONMENT=cache/test-envs/embedding uv sync --python 3.12 --locked --extra embedding --group dev --cache-dir cache/uv
cache/test-envs/embedding/bin/python -m pytest tests/embedding tests/common tests/architecture -q

UV_PROJECT_ENVIRONMENT=cache/test-envs/koblex uv sync --python 3.12 --locked --extra koblex --group dev --cache-dir cache/uv
cache/test-envs/koblex/bin/python -m pytest tests/koblex tests/common tests/architecture -q
cache/test-envs/koblex/bin/python -m llm_benchmark.koblex
```

로컬에서는 `.venv/bin/uv`를 사용할 수도 있습니다. KoBLEX 전용 환경에는 NumPy·Torch·MTEB를 설치하지 않습니다. 임베딩 extra는 MTEB의 전이 의존성으로 PyArrow를 설치할 수 있지만 KoBLEX 코드를 호출하지 않습니다. 새 라이브러리는 필요한 extra에만 추가합니다. 하나의 lockfile은 공유하므로 의존성 버전 충돌까지 완전히 분리한 프로젝트는 아닙니다.

테스트는 `--import-mode=importlib`로 수집해 파일명이 겹쳐도 충돌하지 않습니다. 각 실험의 fixture는 자기 `conftest.py`에 두고 다른 테스트 파일을 import하지 않습니다. 임베딩 테스트의 MTEB 캐시와 생성 데이터·결과는 pytest 임시 폴더를 사용하고, 환경변수는 테스트 후 복구합니다. 공통·다른 실험의 모듈을 우회해서 가져오는 변경은 의존 경계 테스트가 검출합니다.

전체 검증은 두 extra가 설치된 환경에서 `python -m pytest -q`로 수행합니다. 선택한 extra만 있는 환경에서는 다른 실험의 동적 import 검사가 해당 extra 미설치 사유로 skip될 수 있습니다.

## 코드 읽는 순서

`src/llm_benchmark/embedding/run.py`의 `run_benchmark()`에서 시작합니다. 데이터 처리는 `dataset.py`, 모델과 캐시는 `embeddings.py`, 검색과 지표는 `evaluation.py`에 있습니다.

CLI 인자는 `src/llm_benchmark/embedding/__main__.py`에서 처리합니다. `.venv/bin/python -m llm_benchmark.embedding prepare` 방식으로도 실행할 수 있으며, 기존 `.venv/bin/python -m llm_benchmark.embedding.run prepare`는 호환 진입점으로 유지합니다.

## 데이터 준비

```bash
.venv/bin/embedding-benchmark download
.venv/bin/embedding-benchmark prepare
```

`download`는 고정 리비전에서 README·Corpus·Qrels를 내려받고 SHA-256을 기록합니다. `prepare`는 출처와 파일 해시, ID·텍스트·등급을 검증하고 `data/prepared/eve_legal/`에 `corpus.json`, `queries.json`, `qrels.json`, `validation.json`을 저장합니다.

고유 질의 ID는 원문 질의 텍스트를 JSON 인코딩한 SHA-256 앞 24자리로 생성합니다. 질의 순서는 ID 오름차순입니다. 동일 질의가 여러 행에 있으면 설정의 `merge_max`에 따라 관련 문서 합집합과 최대 등급을 사용하고 검증 보고서에 병합 전 등급을 남깁니다. `error`로 변경하면 중복에서 중단합니다.

## 실제 모델 추론 확인

```bash
.venv/bin/embedding-benchmark probe
.venv/bin/embedding-benchmark probe --models multilingual-e5-large
```

각 모델에 실제 문서 2개와 질의 2개를 입력해 벡터 생성·길이 통계·실제 Pooling을 확인합니다. `results/probes/`의 보고서는 추론 동작 확인용이며 Recall 점수가 아닙니다.

## 전체 벤치마크

```bash
.venv/bin/embedding-benchmark run
```

세 모델을 순차 실행합니다. 특정 모델만 실행하거나 장치를 지정할 수 있습니다.

```bash
.venv/bin/embedding-benchmark run --models multilingual-e5-large --device mps --batch-size 4
```

`auto`는 CUDA, MPS, CPU 순으로 사용 가능한 장치를 선택합니다. 배치 크기와 장치 변경은 결과 manifest와 캐시 키에 반영합니다. 실제 모델 실행은 의존성 설치나 단위 테스트보다 오래 걸립니다.

```bash
.venv/bin/embedding-benchmark run --models multilingual-e5-large --query-limit 10
```

`--query-limit`은 질의 수만 제한합니다. 검색 Corpus는 59,786개 전체를 유지하므로 최초 실행의 문서 임베딩 비용은 줄지 않습니다. 이 실행은 `query_subset`으로 표시하고 전체 평가 결과와 구분합니다.

## 결과와 실패 처리

`results/<run_id>/`에 원본 평가 JSON, 질의 상세 JSON, CSV, manifest를 저장합니다. 완료한 모델의 결과는 다음 모델이 실패해도 남깁니다. 실패 또는 키보드 중단은 manifest에 기록합니다. 강제 프로세스 종료에서는 마지막 상태가 `running`으로 남을 수 있습니다.

문서·질의 벡터는 내용·모델 리비전·환경별 캐시에 저장하고, 재사용할 때 파일 체크섬을 확인합니다. 각 문서/질의 집합의 인코딩이 완료되어야 캐시가 저장됩니다. 인코딩 중단 시 해당 집합은 처음부터 다시 처리합니다.

MTEB 2.26.11의 RetrievalEvaluator가 사용하는 지표 함수에 같은 정확 검색 결과를 직접 전달합니다. 이 연결은 `evaluation.py`에서 관리하며 별도 Custom Task를 생성하지 않습니다. MTEB 지표와 프로젝트의 비반올림 Recall을 비교하고 오차가 반올림 범위를 벗어나면 실패합니다. MTEB의 부가 지표는 원본 JSON에 보존되지만 1차 비교표에는 Recall만 사용합니다.

## 검증

개발 의존성의 Ruff로 작은따옴표·줄바꿈·import 정렬을 확인하고, mypy는 설치된 Python 환경을 기준으로 전체 `src/llm_benchmark`를 검사합니다. 개별 실험을 추가해도 검사 범위를 해당 실험 하나로 바꾸지 않습니다. 모든 실험의 타입 검사에는 해당 extra의 의존성을 설치합니다.

```bash
.venv/bin/ruff check src tests scripts docs/posts/scripts
.venv/bin/ruff format --check src tests scripts docs/posts/scripts
.venv/bin/mypy
```

```bash
.venv/bin/python -m pytest -q
```

테스트는 외부 모델 다운로드 없이 수식, 전체 벡터 검색, 동점, 질의 병합, 캐시 무결성, 실제 MTEB 연동, 결과 파일과 실패 기록을 검증합니다. 테스트에서 만든 작은 벡터는 실제 모델 성능으로 보고하지 않습니다.
