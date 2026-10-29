# 개발 아키텍처

법률 RAG 연구를 위해 임베딩 검색 실험과 평가 데이터 구축을 관리합니다. 상위 패키지는 `llm_benchmark`이며, 임베딩 실험과 KoBLEX 평가 구축 코드를 역할에 따라 구분합니다.

## 코드 구조

```text
src/llm_benchmark/
  common/
    hashing.py       # 파일 SHA-256·기존 JSON 해시 형식 유지
    io.py            # 원자적 JSON·JSONL 읽기/쓰기
  embedding/
    __main__.py      # CLI 인자 처리와 명령 분기
    config.py        # YAML 입력 검증과 설정 구성
    models.py        # 원본 스키마·내부 데이터·단계별 결과 타입
    parsing.py       # 외부 입력의 자료형 검증
    dataset.py       # EVE 다운로드, 데이터 검증, 정규화, JSON 저장
    embeddings.py    # 모델 로드, 입력 구성, 임베딩, 캐시 검증
    evaluation.py    # 전체 Corpus 정확 검색, Recall 계산, MTEB 검증
    run.py           # 실험 순서, probe, 결과 저장
  koblex/
    models.py        # 원본·매핑·분할·평가 데이터 타입
    download.py      # 고정 원본 다운로드·해시 검증
    prepare.py       # 원본·매핑·분할 단계 객체 연결과 저장
    mapping.py       # 법령 정규화·근거 매핑·정답 집합
    splitting.py     # 유사 질문 그룹화·평가 분할
    audit.py         # 무결성 검증·실행 통계
    evaluation.py    # 근거 회수 지표·원문 구간 검증
    __main__.py      # KoBLEX 구축 실행 진입점
```

`embedding/run.py`의 `run_benchmark()`부터 읽으면 임베딩 실험 전체를 따라갈 수 있습니다. `embedding-benchmark` 명령은 `embedding/__main__.py`의 `main()`을 실행합니다. 기존 `embedding.run` 직접 실행은 새 CLI에 위임합니다. KoBLEX는 `python -m llm_benchmark.koblex`로 별도 실행하며 모델을 로드하지 않습니다. `prepare()`는 `load_sources → map_sources → partition_dataset → 검증·저장` 순서로 실행하며, 각 단계는 `SourceDataset → MappedDataset → PreparedDataset` 객체로 연결합니다. 원본 읽기·저장만 파일에 접근합니다.

실험 내부는 상대 import, 공통 모듈은 절대 import를 사용합니다. 새로 정의하거나 수정하는 내부 dataclass는 `slots=True, kw_only=True`를 기본으로 하고 원본 필드는 TypedDict로 명시합니다. 외부 입력 검증과 기존 파일·캐시의 직렬화 형식 및 필드 순서를 보존합니다.

의존 방향은 `embedding → common`, `koblex → common`입니다. 두 실험은 서로 import하지 않고, `common`도 실험 코드를 import하지 않습니다. `common`은 표준 라이브러리만 사용하며 법률 정답·청킹·검색 지표·모델 설정은 각 실험에 둡니다. JSON 해시는 기존 EVE 캐시와 KoBLEX ID가 바뀌지 않도록 두 직렬화 형식을 명시적으로 유지합니다.

각 실험의 설치 의존성도 `embedding`/`koblex` extra로 구분합니다. 테스트는 별도 환경에서 실행할 수 있고, 실험 fixture·임시 데이터·캐시는 다른 테스트에 의존하지 않습니다. `tests/architecture/`가 소스와 테스트의 교차 import 및 새 프로세스에서 다른 실험 없이 import되는지를 검사합니다. 공통 변경은 `tests/common/`과 각 소비 실험 테스트로 함께 검증합니다.

2026-10-23 검증에서 전체 34개, 임베딩 전용 환경 26개가 통과했습니다. KoBLEX 전용 환경은 16개 통과·임베딩 extra 미설치에 따른 검사 1개 skip이며 데이터 구축도 완료했습니다. 공통 코드 분리 전후 KoBLEX 산출물 7개의 해시는 동일하고, 잠금 파일의 외부 라이브러리 버전도 바뀌지 않았습니다.

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
tests/koblex/             KoBLEX 데이터·평가 회귀 테스트
tests/common/             공통 저장·해시 호환성 검증
tests/architecture/       실험 간 의존 경계 검증

docs/engineering/         개발 기술과 실행 방법
docs/domain/legal/        법률 데이터와 정답 규칙
docs/benchmarks/          실험 목록·개별 안내·평가 기준·결과
docs/prompts/             시스템 프롬프트
docs/references/          제공 원본 문서
docs/work-log/            로컬 전용 수행 기록 (Git 추적 제외)
```

자세한 명령은 [실행 방법](running.md), 입력 규칙은 [모델 설정](models.md), 법률 데이터 규칙은 [데이터셋 문서](../domain/legal/dataset.md)를 참조합니다.
