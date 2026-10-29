# 문서 목차

[프로젝트 소개](../README.md) · [전체 실험 목록](benchmarks/README.md)

## 실험 문서

실험은 [실험 목록](benchmarks/README.md)에 등록하고 개별 문서에서 관리합니다.

- [한국어 판례 임베딩 검색 비교](benchmarks/embedding/README.md)
- [KoBLEX 법령·정답 근거 구축](benchmarks/koblex/README.md)
- [새 실험 문서 양식](benchmarks/experiment-template.md)

평가 조건과 지표는 각 실험의 설계 문서에, 실제 실행 수치와 한계는 결과 보고서에 기록합니다. 법률 분야의 후속 실험 방향은 [법률 RAG 연구 계획](research-plan.md)에서 관리합니다.

## 공통 개발 문서

- [개발 아키텍처](engineering/architecture.md)
- [설치·실행 환경과 실험별 격리](engineering/running.md)
- [Git 커밋·브랜치 규칙](engineering/git-workflow.md)
- [주석과 코드 설명 작성 규칙](../AGENTS.md)
- [시스템 프롬프트 관리](prompts/README.md)

## 분야별 자료

- [법률 데이터셋과 정답 관리](domain/legal/dataset.md)
- [법률 데이터 확인 기록](domain/legal/data-inspection-2026-10-19.md)
- [임베딩 모델 입력 규칙과 리비전](engineering/models.md)
- [참고 자료 안내](references/README.md)
- [블로그 원고 목록](posts/README.md)

## 폴더 역할

| 폴더 | 관리 내용 |
| --- | --- |
| `benchmarks/` | 실험 목록, 실험별 안내, 평가 설계와 결과 보고서 |
| `engineering/` | 언어, 의존성, 모듈 구조, 실행 환경 |
| `domain/` | 분야별 데이터 규칙과 해석 |
| `prompts/` | 용도별 프롬프트와 버전 관리 |
| `posts/` | 공유용 글과 블로그 원고 |
| `references/` | 제공받은 참고 자료의 원본 사본 |
| `work-log/` | 로컬 전용 수행 기록, Git 추적 제외 |

참고 원본과 날짜별 결과 보고서는 기존 경로에 보존합니다. 현재 기준은 실험별 안내와 설계 문서에서 확인하고, 작업 과정은 로컬 기록으로 관리합니다.
