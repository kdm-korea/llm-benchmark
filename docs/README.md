# 문서 구조

개발 기술, 평가 방법, 법률 도메인, 제공 자료와 실제 작업 기록을 분리합니다.

| 폴더 | 관리 내용 |
| --- | --- |
| `engineering/` | 언어, 의존성, 모듈 구조, 실행 환경 |
| `benchmarks/` | 평가 단위, 지표, 실험 통제, 완료 기준 |
| `domain/legal/` | 법률 데이터셋, 식별자, Qrels 해석 |
| `prompts/` | 용도별 시스템 프롬프트와 버전 관리 |
| `references/` | 제공받은 참고 자료의 원본 사본 |
| `work-log/` | 실제 변경, 검증 결과, 미완료 사항 |

## 기준 문서

- [개발 아키텍처](engineering/architecture.md)
- [Git 커밋·브랜치 규칙](engineering/git-workflow.md)
- [실행 방법](engineering/running.md)
- [모델 설정과 근거](engineering/models.md)
- [실제 데이터 확인 기록](domain/legal/data-inspection-2026-10-19.md)
- [Recall@K 평가 설계](benchmarks/embedding-recall.md)
- [법률 데이터셋](domain/legal/dataset.md)
- [시스템 프롬프트 관리](prompts/README.md)
- [참고 자료 안내](references/README.md)
- [작업 기록](work-log/README.md)

설계 문서는 현재 기준을 설명하고 작업 기록은 실제 이력을 보존합니다. 초기 기록의 청크 단위 평가와 MD 원문 가정은 제공된 계획서를 확인한 뒤 문서 단위 평가와 공개 데이터셋 사용으로 변경했습니다. 참고 원본은 수정하지 않고, 변경된 설계는 현재 문서와 작업 기록에 반영합니다.
