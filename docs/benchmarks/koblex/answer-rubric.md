# KoBLEX 답변 검수표 v1

- 작성일: 2026-10-23 (Asia/Seoul)
- 상태: 평가 양식 준비. 생성·LLM judge 실행 없음. 합격선·가중합·judge 모델은 미확정.
- 평가 대상은 고정 스냅샷의 시나리오·질문, 원본 참고 답변, 실제 전달 근거다. 현재 법률상 결론을 검증했다는 뜻은 아니다.

## 1. 분리할 생성 조건

| 조건 | 입력 | 목적 |
| --- | --- | --- |
| `oracle_evidence` | 시나리오+질문, 주석된 정답 근거 전체 | 검색이 충분할 때 생성의 오류 확인 |
| `retrieved_evidence` | 같은 시나리오+질문, 실제 검색·예산 절단 이후 근거 | 검색부터 답변까지 평가 |

동일 모델·프롬프트·디코딩 조건·토큰 예산을 고정해 비교한다. 참고 답변은 생성 입력에 제공하지 않는다. judge는 참고 답변과 정답 근거를 볼 수 있으나 생성기가 실제 받은 근거와 구분해서 받는다. 보류 질문은 주 결과에서 제외한다.

## 2. 차원별 0/1/2점과 근거

| 차원 | 0점 | 1점 | 2점 |
| --- | --- | --- | --- |
| 결론 정확성 | 질문의 핵심 결론·당사자·기간·금액 등이 틀림 | 핵심 방향은 맞지만 유의미한 세부 오류 또는 불명확성 | 스냅샷과 참고 답변을 기준으로 핵심 결론·수치·당사자를 정확히 설명 |
| 질문 충족·완전성 | 핵심 하위 질문을 대부분 답하지 못함 | 일부 하위 질문·중요 조건·예외 누락 | 모든 하위 질문과 중요한 요건·단서·예외를 충족 |
| 제공 근거 충실성 | 실제 전달 근거와 모순되거나 핵심 주장이 근거 없음 | 일부 주장이 근거를 과도하게 확대하거나 지지가 불충분 | 모든 실질적 주장이 전달 근거에 의해 뒷받침되며 추론/미확인 사항을 구분 |
| 인용 존재·정확성 | 인용이 허구이거나 핵심 근거 연결이 틀림 | 일부 위치·조항 번호가 틀리거나 필요한 인용이 누락 | 법령·조/항/호·unit/span이 존재하고 해당 주장을 실제로 지지 |
| 복수 조문 적용 | 핵심 조문 관계를 잘못 적용 | 일부 관계만 적용하거나 요건 결합·예외 적용이 불완전 | 필요한 조문 간 관계를 사실관계에 맞게 적용하고 이유를 설명 |
| 정보 부족 대응 | 근거·사실이 부족한데 단정하거나 답을 꾸밈 | 불확실성을 언급하나 필요한 유보·추가 질문이 불완전 | 실제 부족한 사실·근거·시점을 특정하고 적절히 유보·질문 |

해당 없음은 `null`과 사유를 넣는다. 단일 조문 질문의 복수 조문 적용, 충분한 근거가 제공된 질문의 정보 부족 대응 등이 해당할 수 있다. `null`을 0 또는 만점으로 합산하지 않는다. 점수마다 구체적 주장·참고 답변 또는 근거 위치·판정 이유를 기록한다. 참고 답변의 표현과 문자상 다르다는 이유만으로 감점하지 않는다.

원본 정답의 오류·누락 의심이 있으면 `reference_disputed`로 분리하고 검토 전까지 결론 차원의 자동 합격/불합격 판정을 유보한다. 원본 밖의 타당한 근거를 모델이 인용했다면 미판정으로 보내고 즉시 환각으로 단정하지 않는다. 원본이 필수/보조 근거를 구분하지 않으므로 모든 context를 답변에 인용하지 않았다는 이유만으로 결론 오류를 부여하지 않는다. 검색의 엄격한 All-evidence Hit과 답변의 필요 근거 충족을 구분한다.

## 3. 판정 기록 계약

후속 결과 한 행에는 다음을 남긴다.

```json
{
  "query_id": "원본 ID",
  "split": "dev 또는 test",
  "condition": "oracle_evidence 또는 retrieved_evidence",
  "source_manifest_sha256": "...",
  "mapping_version": "koblex-v1",
  "generation": {"model": null, "revision": null, "prompt_sha256": null, "parameters": {}},
  "retrieval": {"run_id": null, "ranked_chunk_ids": [], "delivered_spans": [], "token_budget": null},
  "answer": "...",
  "reviewer": {"type": "human 또는 llm", "id": null, "model_revision": null},
  "scores": {
    "correctness": null, "completeness": null, "faithfulness": null,
    "citation_accuracy": null, "multi_provision_application": null, "insufficient_information": null
  },
  "reasons_by_dimension": {},
  "unsupported_claims": [],
  "citation_checks": [],
  "reference_disputed": false,
  "error_types": [],
  "review_status": "pending"
}
```

오류 유형은 `retrieval_missing`, `generation_wrong_conclusion`, `missing_exception`, `wrong_party_or_number`, `unsupported_claim`, `invalid_citation`, `temporal_unknown`, `reference_disputed` 등을 병기할 수 있다. 검색에 없는 내용을 생성이 맞혔더라도 충실성과 결론을 별도로 평가한다. oracle 조건도 틀리면 생성·정답 기준 문제를 먼저 검토한다. oracle은 맞고 retrieved 조건만 틀리면 검색 누락과 전달 예산을 함께 검토한다. 이 두 조건만으로 완전한 인과 분해를 했다고 주장하지 않는다.

자동 평가 도입 시 dev에서 단일/복수 근거·예외·유사 조문·정보 부족 사례를 포함한 사람 검수 표본을 먼저 정하고, judge의 점수 편향과 불일치를 기록한다. 독립 검수자 2인이 가능한 표본에는 각각 판정 후 불일치 조정 이력을 남긴다. 이번 데이터 매핑 표본 점검은 이 법률 전문가 답변 검수를 대체하지 않는다.

## 4. LF-Eval 검토 결과와 도입 결정

[공식 구현 고정본](https://github.com/daehuikim/KoBLEX/blob/1a2d225be33a70ebf91e2b6369acf27551dcfbb4/experiments/lf-eval/eval_pipeline.py)을 내려받아 읽었다. 원본 파일 SHA-256은 source manifest에 있다. 이번에는 실행하지 않았다.

- DeepEval `GEval`에 질문(시나리오+질문), 생성 답변, 참고 답변, 정답 context를 넣는다.
- 지침은 법적 정확성·주석된 조문 활용·참고 답변 세부 일치 등을 평가하며 1~10점으로 표현돼 있다. 실제 점수 척도·정규화는 설치될 DeepEval 동작을 검증하기 전 확정하지 않는다.
- judge 모델·모델 리비전·DeepEval 버전·평가 임계값이 코드에서 고정돼 있지 않다. 설치 버전과 기본값에 결과가 의존할 수 있다.
- README의 예시 `answers` 구조와 실제 코드의 `answer`/`provisions` 입력이 다르다. 실제 코드는 답변을 최대 800문자로 자르고 특수 think 구분자로 split한다. 전처리로 내용이 손실될 수 있다.
- 원본 gold/prediction을 질문 ID로 join하지 않고 순서대로 `zip`한다. 누락·행 순서 차이를 검출하지 못하므로 그대로 사용하면 비교 대상이 어긋날 수 있다.
- retrieval 평가는 hierarchy+content 문자열 집합의 일치를 사용한다. 이번 다중 행 근거 매핑·문자 구간 포함 규칙과 다르다.
- 정답 context를 judge에 주므로 실제 전달 근거만의 충실성 점수를 직접 대체하지 않는다. 정보 부족에도 일반 답변을 하는 반응을 강하게 감점하는 기준은 별도 실패 대응 평가와 맞춰 검토해야 한다.

따라서 **LF-Eval은 후속 비교용 후보로 보류**한다. 채택 전 ID join·길이 검증, 전체 답변 보존, judge/의존성/전처리/척도 고정, 실제 전달 context와 oracle context 구분, 비용 확인을 거친다. 그 조건을 기록하고 사람 검수와 일치도를 확인한 뒤 별도 점수로 보고한다. 원본 LF-Eval 점수 재현이나 API 비용 실험을 이번 단계의 완료 내용에 포함하지 않는다.
