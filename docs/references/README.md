# 제공 참고 자료

[legal_embedding_recall_benchmark.md](legal_embedding_recall_benchmark.md)는 2026-10-19 사용자가 제공한 벤치마크 계획서를 내용 변경 없이 보관한 사본입니다.

- 원본 경로: `/Users/kimdongmin/Downloads/legal_embedding_recall_benchmark.md`
- 용도: 실험 범위, 데이터셋, 비교 모델, 지표와 산출물 설계의 근거
- 성격: 검색 Corpus가 아닌 계획 문서

계획서의 디렉터리 예시는 승인된 개발 구조와 법률 도메인 분리 원칙에 맞춰 조정했습니다. Python 상위 패키지는 `llm_benchmark`, 의존성 관리는 `pyproject.toml`과 uv를 사용하도록 설계했습니다. 의존성은 원본의 `requirements.txt` 대신 `pyproject.toml`과 `uv.lock`으로 관리합니다. 2026-10-20 임베딩 코드를 네 모듈로 단순화했고, 2026-10-23 `embedding/` 실험 폴더와 `common/` 유틸리티 및 독립 테스트 구조로 정리했습니다.

원본을 처음 보관한 시점에는 문서만 개정했습니다. 이후 데이터 다운로드, 구조 확인, 코드 구현과 모델 추론 확인을 진행했습니다. 실행 이력은 [작업 기록](../work-log/README.md)에 보존하며, 제공 원본 자체는 변경하지 않습니다.
