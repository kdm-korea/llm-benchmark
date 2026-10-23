# 블로그 포스팅 원고

`2026-10-23-legal-embedding-recall-benchmark.md`는 Jekyll 포스트 원고입니다.

- [게시된 글](https://kdm-korea.github.io/rag/embedding/2026/10/23/legal-embedding-recall-benchmark.html)
- [현재 실험 안내](../benchmarks/embedding/README.md)
- [게시 링크·지표 검증 기록](2026-10-23-link-audit.md)

## 게시 파일 배치

| 이 저장소 | 블로그 저장소 |
| --- | --- |
| `docs/posts/2026-10-23-legal-embedding-recall-benchmark.md` | `_posts/2026-10-23-legal-embedding-recall-benchmark.md` |
| `assets/images/legal-embedding-recall/metrics-overview.png` | `assets/images/legal-embedding-recall/metrics-overview.png` |
| `assets/images/legal-embedding-recall/recall-curve.png` | `assets/images/legal-embedding-recall/recall-curve.png` |
| `assets/images/legal-embedding-recall/ndcg-curve.png` | `assets/images/legal-embedding-recall/ndcg-curve.png` |

이미지 주소는 Jekyll `relative_url` 필터를 사용합니다. 원고와 이미지 세 장을 함께 옮겨야 합니다. 전체 Jekyll 사이트 빌드와 배포 검증은 이 원고의 로컬 검증에 포함하지 않습니다.

## 그래프 재생성과 검증

`matplotlib`이 설치된 프로젝트 가상환경에서 실행합니다. 모델 추론은 다시 실행하지 않습니다.

```sh
MPLCONFIGDIR=/private/tmp/legal-mpl .venv/bin/python scripts/render_benchmark_figures.py
```

원본 전체 실행 결과와 준비된 Qrels가 로컬에 있어야 합니다. 해당 데이터는 Git 추적 대상이 아닙니다.

그래프 생성 시 다음을 확인합니다.

1. Recall의 질의별 저장 점수 및 MTEB 평균을 원본 CSV와 대조하고, nDCG는 MTEB 질의별 평균과 원본 집계를 대조합니다.
2. 전체 질의의 검색 순위와 준비된 정답 집합의 교집합으로 Recall과 이진 nDCG의 DCG/IDCG를 다시 계산합니다.
3. 그래프의 실제 좌표와 표시 라벨을 검증한 수치와 대조합니다.

수치·원본 파일 해시·이미지 해시는 `assets/images/legal-embedding-recall/verification.json`에 저장합니다. 본문 표는 별도로 CSV·manifest·설정·질의별 결과 및 MTEB와 두 차례 대조하였습니다. 입력 잘림 비율과 분 단위 시간은 정수 개수와 초 단위 시간에서 별도로 다시 계산하였습니다.

표·문구·이미지 경로와 원본 해시의 재검증은 `.venv/bin/python scripts/verify_benchmark_documents.py`로 실행합니다. nDCG 표는 0~1 점수의 소수점 다섯 자리이며, Recall 표와 그래프는 백분율의 소수점 둘째 자리입니다.
