# EVE 데이터 확인 기록: 2026-10-19

## 출처와 고정 리비전

[공식 데이터셋](https://huggingface.co/datasets/ysmeta/EVE-Bench-Legal-2.0)의 API, README와 실제 JSONL을 확인했습니다.

- 리비전: `1580448204d13ec8af13745086ffa91cfda407c3`
- Corpus: 59,786행
- Qrels: 5,999행, 고유 질의 5,998개
- 관련 문서가 없는 고유 질의: 0개
- 질의별 관련 문서 수: 최소 1, 최대 35, 평균 약 7.1962
- 평가 기준: `grades > 0`, 중복 질의 병합 후 집계

데이터셋 카드의 `test` 뷰어는 `eval_synth`를 표시합니다. 이번 평가는 `eval_synth`의 단일 gold 대신 `qrels.jsonl`의 실제 질의와 다중 관련성 판정을 사용합니다. 내부 분할 이름은 `test`로 관리합니다.

## 실제 필드

- `corpus.jsonl`: `docid`, `title`, `text`, `kind`
- `qrels.jsonl`: `query`, `gold`, `pool_size`, `grades`
- `grades`: 문서 ID를 키로 하고 관련성 등급을 값으로 하는 객체
- 카드 설명상 등급 2는 원래 gold, 1은 추가 관련 문서

전체 검색 Corpus의 제목과 text를 줄바꿈으로 연결합니다. 제목·본문 문자열과 원본 문서 ID를 보존합니다. 해당 판례 텍스트를 판결문 전문 전체라고 가정하지 않습니다.

## 중복 질의 발견과 정책

동일한 세금 관련 질의가 서로 다른 gold를 가진 두 행에 존재합니다. `query`만 키로 만들어 뒤 행으로 덮어쓰면 일부 관련 문서와 등급을 잃습니다.

기본 설정은 `duplicate_query_policy: merge_max`입니다. 같은 질의는 한 번 평가하고, 관련 문서는 합집합, 문서별 등급은 최댓값을 사용합니다. 병합 전후를 검증 보고서와 실행 manifest에서 추적합니다. 이 선택은 원본 카드의 단순 dict 구성 예제와 달라 공식 수치를 그대로 재현하는 설정으로 간주하지 않습니다.

문서 ID 중복이나 누락된 Qrels 문서, 유효하지 않은 등급은 병합으로 무시하지 않고 오류로 처리합니다.

## 파일 체크섬

| 파일 | SHA-256 |
| --- | --- |
| corpus.jsonl | `50674d5e2797982534c40a34cbbaef75dde0af80b8ea54e6d11caf4d537eabef` |
| qrels.jsonl | `3418ccb9e5d352843e5a055bc0db14df475900214806fa5232e6ce6005eb1aa4` |

원본은 `data/raw/eve_legal/`, 정규화 데이터와 검증 결과는 `data/prepared/eve_legal/`에 저장했습니다. 원본과 대용량 파생 데이터는 Git 추적에서 제외했습니다.

## 해석 한계

Qrels는 후보 풀의 판정에 근거하므로 전체 Corpus의 모든 관련 문서를 포괄한다고 볼 수 없습니다. 미판정 문서는 Recall 정답 집합에 포함되지 않지만 비관련성이 확정된 것으로 설명하지 않습니다. 데이터셋 카드의 라이선스 표기는 OpenRAIL이며 원본 README도 함께 보관합니다.
