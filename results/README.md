# 실행 결과

- `<run_id>/manifest.json`: 설정, 모델·데이터 리비전, 환경, 평가 범위와 상태
- `<run_id>/raw/`: 모델별 MTEB 원본 평가 결과
- `<run_id>/per_query/`: 질의별 검색 순위·점수·Recall과 제외 사유
- `<run_id>/recall_summary.csv`: 실행 내 모델별 Recall 비교
- `probes/`: 소량 실제 입력의 추론 동작 확인, 벤치마크 점수 아님

실행 결과와 캐시는 Git에서 제외하며 이 안내 문서만 추적합니다. `status: complete`는 해당 실행 범위의 완료를 의미합니다. `mode: query_subset`이나 추론 probe는 전체 벤치마크 완료로 해석하지 않습니다.
