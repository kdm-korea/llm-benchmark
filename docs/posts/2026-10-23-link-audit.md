# 게시 글 링크·참조 검증

- 확인일: 2026-10-23 (Asia/Seoul)
- 대상: [한국어 판례 검색을 위한 임베딩 모델의 Recall과 nDCG 비교](https://kdm-korea.github.io/rag/embedding/2026/10/23/legal-embedding-recall-benchmark.html)
- 결과: 확인한 게시 URL·이미지·본문 출처에서 깨진 링크나 지표 불일치를 발견하지 못했습니다.

## 확인한 항목

| 항목 | 결과 |
| --- | --- |
| 사이트맵과 게시 URL | HTTP 200, 사이트맵에서 실제 게시 주소 확인 |
| 그래프 3장 | 모두 HTTP 200, 로컬 이미지와 SHA-256 일치 |
| 본문 출처 링크 6개 | 데이터셋·모델 3개·MTEB·구현 커밋 모두 HTTP 200 |
| 배너 단축 링크 | 이미지 URL로 이동 후 HTTP 200 |
| 게시 HTML의 Recall·이진 nDCG 표 | 지표 36개가 기존 검증 자료와 일치 |
| 원고 동기화 | 로컬 원고와 블로그 master의 Git blob SHA 일치 |
| 고정 구현 참조 | 참조 커밋의 데이터·모델 리비전과 입력 설정이 실험 기록과 일치 |

블로그 master의 원고 blob SHA는 `1121d8afeee2c23fa3dbf73232a275ebcc3ae977`입니다. 비교한 로컬 원고는 [기존 포스트](2026-10-23-legal-embedding-recall-benchmark.md), 지표·이미지 검증 자료는 [verification.json](../../assets/images/legal-embedding-recall/verification.json)입니다.

## 문서 이동의 영향

블로그는 구현을 [고정 커밋 `2bc01e7`](https://github.com/kdm-korea/llm-benchmark/tree/2bc01e7fbb2c53619b5b0535ed21bfc497178128)으로 참조합니다. 현재 README와 Python 패키지 폴더를 정리해도 이 과거 구현 참조는 유지됩니다. 당시 구현을 재현하는 용도이므로 최신 브랜치 링크로 바꾸지 않았습니다.

현재 프로젝트 안내는 [루트 README](../../README.md), 개별 실험의 최신 안내와 상세 결과는 [판례 임베딩 검색 문서](../benchmarks/embedding/README.md)에서 확인합니다.

문서·링크 검증만 수행했습니다. 모델 추론, 새 성능 실험, 블로그 원고 수정이나 재배포는 하지 않았습니다. HTTP 응답은 확인 시점 기준이며 외부 문서의 모든 내용과 링크의 영구적인 유효성을 보장하는 검사는 아닙니다.
