# Git 작업 규칙

## 커밋 형식

```text
<type>(<scope>): <변경 내용>
```

scope는 필요할 때만 사용하고 설명은 한국어로 작성합니다. 최종 이력에서는 기능별로 하나의 커밋을 남기고, 해당 기능의 코드·테스트·문서를 함께 포함합니다.

| type | 용도 |
| --- | --- |
| feat | 기능 추가 |
| fix | 오류 수정 |
| refactor | 동작을 유지하는 구조 개선 |
| perf | 처리 속도·메모리 개선 |
| test | 테스트 추가·수정 |
| docs | 문서 변경 |
| chore | 의존성·개발 설정 관리 |
| res+ | 리소스 추가 |
| res- | 리소스 삭제 |
| res^ | 리소스 변경 |

scope 예시는 `dataset`, `embeddings`, `evaluation`, `run`입니다. res 타입은 데이터 파일·평가 기준 자료 등 리소스 자체의 변경에 사용합니다. 리소스를 처리하는 코드 변경은 feat, fix 등의 해당 타입을 사용합니다.

```text
feat(embeddings): 임베딩 중간 저장과 재시작 지원
perf(embeddings): 배치 처리량 개선
res+(dataset): 평가 질의 리소스 추가
res-(dataset): 사용하지 않는 평가 리소스 삭제
res^(dataset): 정답 연결 리소스 변경
```

커밋 타입에 리소스 변경을 기록하더라도 `.gitignore`로 제외한 원본 데이터·모델·캐시·실행 결과를 자동으로 Git 추적에 포함하지 않습니다.

## 브랜치

검증된 코드는 `main`에서 관리합니다. 작업 브랜치는 main에서 생성하고 한 브랜치에서 하나의 기능 또는 독립적인 변경 목적을 처리합니다. 서로 다른 기능은 브랜치와 PR을 분리합니다. 이름에는 영문 소문자·숫자·하이픈을 사용하며 `/`는 사용하지 않습니다.

```text
feat-embedding-checkpoint
fix-recall-calculation
refactor-evaluation
perf-embedding-batch
test-recall-edge-cases
docs-git-workflow
chore-update-dependencies
```

브랜치명은 `<type>-<작업명>` 형식을 사용합니다. `res+`, `res-`, `res^`는 커밋 메시지 전용 타입이며 브랜치 규칙에 추가하지 않습니다.

## 병합

기능별로 **작업 브랜치 하나 → PR 하나 → main의 최종 커밋 하나**를 원칙으로 합니다.

작업 중에는 진행 상황을 저장하기 위해 여러 커밋을 만들 수 있습니다. 변경과 테스트를 완료한 뒤 PR을 생성하고 Squash merge하여 해당 기능을 최종 커밋 하나로 합칩니다. PR 제목과 최종 커밋 제목은 커밋 규칙을 따릅니다.

기능에 필요한 코드·테스트·문서는 같은 최종 커밋에 포함합니다. 독립적인 버그 수정, 리팩터링, 성능 개선, 문서 또는 리소스 작업도 변경 목적별로 같은 원칙을 적용합니다.

병합한 작업 브랜치는 삭제합니다. 초기에는 별도 develop·release 브랜치 없이 운영합니다.
