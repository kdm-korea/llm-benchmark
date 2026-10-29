# KoBLEX 법령·정답 근거 구축

[전체 실험 목록](../README.md)

- 상태: 평가 기반 v1 구축 완료. 임베딩·법령 검색·답변 생성 성능 평가는 미실행입니다.
- 목적: 법률 RAG에서 필요한 조문을 찾았는지, 해당 근거로 적절한 답변을 했는지 평가할 데이터와 기준을 준비합니다.
- 구현: `src/llm_benchmark/koblex/`
- 테스트: `tests/koblex/`

## 데이터 흐름과 실행 결과

- [데이터 흐름·스키마·조인과 매핑 예시](data-flow.md): 원본부터 평가 정답까지 연결 구조
- [2026-10-24 재실행 결과](results-2026-10-24.md): 실제 생성 개수·검증·산출물 해시

## 데이터와 완료 범위

KoBLEX QA 226개와 법령 233,544행을 고정 리비전으로 확보했습니다. 정확히 같은 법령 행만 합쳐 209,564개 단위로 정리했고, 질문별 원본 근거를 연결했습니다.

222문항은 평가에 사용할 수 있으며 4문항은 근거 불일치로 보류했습니다. 평가 가능 질문은 dev 44개·test 178개로 나눴습니다. 이는 정답 매핑 준비 상태이며 법률 전문가가 모든 답변의 정확성을 검수했다는 뜻은 아닙니다.

자세한 스키마·중복·계층·라이선스·시점 확인 결과와 보류 사유는 [데이터 검증 보고서](audit-report.md), 원본 리비전과 해시는 [source manifest](source-manifest.json)에 있습니다.

## 실행 방법

프로젝트 루트에서 실행합니다.

```bash
uv sync --locked --extra koblex --cache-dir cache/uv

# 고정 원본 확보 또는 기존 파일 해시 확인
.venv/bin/python -m llm_benchmark.koblex.download

# 데이터 검증·근거 매핑·분할·검증 보고서 생성
.venv/bin/python -m llm_benchmark.koblex

# 데이터·평가 코드 검증
.venv/bin/python -m pytest tests/koblex -q
```

설정은 [configs/koblex.toml](../../../configs/koblex.toml)에서 관리합니다. `--config`로 다른 설정을, `--raw`로 원본 경로를 지정할 수 있습니다. 구축 명령은 `--output`과 `--audit-dir`도 지원합니다. 설정 안의 상대 경로는 프로젝트 루트 기준이고 CLI 경로는 실행 위치 기준입니다.

위 명령은 데이터와 평가 기준을 구축합니다. 모델 임베딩이나 답변 생성 API를 실행하지 않습니다. 실험별 환경 격리 방법은 [공통 실행 안내](../../engineering/running.md)에 있습니다.

## 평가 기준과 산출물

- [검색 평가 규칙](evaluation-rules.md): 근거 회수 판정, Evidence Recall·Hit·All-evidence Hit·MRR, 청크 구간 및 분할 규칙
- [답변 검수표](answer-rubric.md): 결론·완전성·충실성·인용·복수 조문 적용·정보 부족 대응
- [데이터 검증 보고서](audit-report.md): 실제 점검 결과와 한계
- [초기 구축 계획](../koblex-evaluation-plan.md): 연구 범위와 완료 기준

원본은 `data/raw/koblex/`, 정규화 데이터·qrels·분할은 `data/prepared/koblex-v1/`에 저장합니다. 공유용 검증 통계·매핑 장부·검수 표본은 이 문서 폴더에 있습니다.

## 해석 범위

법령의 구조화된 시행일과 질문의 기준 시점이 없어 현재법에 대한 정답 정확도를 검증했다고 해석하지 않습니다. 필수·보조 근거 구분과 원본 정답의 완전성도 확인되지 않았습니다. 후속 검색·생성 실험에서는 데이터 버전을 고정하고 성능 결과를 별도로 기록합니다.
