# 1차 모델 설정과 근거

후보는 제공된 계획서를 따릅니다. 실제 Recall 점수에 근거한 최종 모델 선택은 전체 실험 이후에 수행합니다.

| 모델 | 고정 리비전 | 최대 토큰 | 질의 / 문서 접두어 |
| --- | --- | --- | --- |
| BAAI/bge-m3 | `5617a9f61b028005a4858fdac845db406aefb181` | 8192 | 없음 / 없음 |
| nlpai-lab/KURE-v1 | `8b418a58414668e75532ed045c22d9ca018ae2b2` | 8192 | 없음 / 없음 |
| intfloat/multilingual-e5-large | `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3` | 512 | `query: ` / `passage: ` |

[BGE-M3 공식 카드](https://huggingface.co/BAAI/bge-m3), [KURE-v1 공식 카드](https://huggingface.co/nlpai-lab/KURE-v1), [E5 공식 카드](https://huggingface.co/intfloat/multilingual-e5-large)를 확인했습니다. SentenceTransformer가 고정 모델의 Pooling 구성을 로드하며, 출력은 L2 정규화하고 cosine으로 검색합니다. ST의 기본 프롬프트가 중복 적용되지 않도록 `prompt=""`로 명시하고 입력 접두어는 설정에서 한 번만 적용합니다.

최대 입력 길이는 모델별 공식 제한을 유지합니다. 모든 모델의 길이를 512로 맞추는 실험과는 조건이 다르며, 실제 문서·질의 잘림 건수와 비율을 결과에 함께 기록합니다. 이 실험은 공식 입력 방식과 길이 제한을 포함한 모델별 검색 성능을 비교합니다.

실제 실행 시 요청 리비전을 Hub API로 다시 확인하고 manifest에 resolved SHA를 기록합니다. 라이브러리 버전은 `uv.lock`에 고정하고 결과에도 저장합니다. 모델의 dense 단일 벡터만 사용하며 sparse 또는 multi-vector 검색은 포함하지 않습니다.


## 실제 추론 확인

2026-10-19 세 모델을 각각 실제 문서 2개·질의 2개로 확인했습니다. 모두 문서/질의 벡터 `(2, 1024)`를 생성했고 실제 Pooling은 BGE-M3·KURE-v1의 CLS, E5의 mean으로 확인했습니다. MPS 장치와 torch.float32를 사용했습니다. 이는 소량 추론 확인이며 Recall 또는 처리 속도 비교 결과가 아닙니다.
