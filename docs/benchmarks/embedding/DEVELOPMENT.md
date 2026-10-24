# 임베딩 코드 수정과 검증

`embedding.models`는 표준 라이브러리만으로 import할 수 있다. 원본 JSON 행과 YAML 설정은 TypedDict로 검증하고, 내부 문서·질의·단계 결과는 dataclass로 전달한다. ID 조회표와 통계·출력 JSON은 딕셔너리를 유지한다.

실행 흐름은 `prepare_dataset` → `_encode_dataset` → `_evaluate_dataset` → `_save_model_results` 순서다. `EncodedDataset`이 문서·질의 벡터와 통계를, `ModelEvaluation`이 질의별 Recall과 MTEB 결과를 전달한다. 모델은 임베딩 단계의 `finally`에서 해제한다.

저장과 해시 계산 경계에서 `corpus_payload`, `query_payload`, `to_dict`로 기존 JSON 구조를 복원한다. 문서 입력은 공백을 정리한 제목과 본문을 개행으로 연결하며, 질의 ID·입력 순서·정수 등급 표현을 유지한다. 캐시 키의 schema version이나 모델·실행 설정 필드를 바꾸면 기존 캐시를 재사용할 수 없다.

```sh
.venv/bin/pytest -q
UV_TOOL_DIR=cache/type-tools UV_TOOL_BIN_DIR=cache/type-tools/bin .venv/bin/uv tool run --offline --cache-dir cache/uv --from mypy mypy --python-executable .venv/bin/python
.venv/bin/python scripts/verify_benchmark_documents.py
.venv/bin/uv lock --check --offline --cache-dir cache/uv
```

mypy 범위는 `embedding`과 `common`이며 캐시는 `cache/mypy`에 둔다. 테스트는 모델 추론을 대체해 네트워크 없이 검색·MTEB 연동·실패/중단 기록을 검사한다. `tests/embedding/fixtures`의 캐시 메타데이터는 dataclass 전환 이전 구현으로 생성한 작은 고정 입력의 호환성 기준이다.
