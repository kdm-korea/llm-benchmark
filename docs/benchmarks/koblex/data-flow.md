# KoBLEX 데이터 흐름

[실험 안내](README.md) · [실행 결과](results-2026-10-24.md)

질문의 정답 근거를 법령 본문에 연결해 정답표와 테스트셋을 생성합니다. 아래는 JSON·JSONL 데이터를 테이블 형태로 표현한 관계도입니다.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#ffffff", "primaryTextColor": "#111827", "primaryBorderColor": "#64748b", "secondaryColor": "#ffffff", "tertiaryColor": "#ffffff", "lineColor": "#475569", "textColor": "#111827"}}}%%
classDiagram
    direction TB
    class QUERIES["질문 · queries"] {
        query_id
        scenario
        question
        retrieval_query
        reference_answer
        evaluation_status
    }
    class EVIDENCE["근거 · evidence"] {
        query_id
        context_position
        evidence_id
        original_context
        status
        unit_ids
        candidate_unit_ids
        reason
    }
    class EVIDENCE_UNIT["근거와 법령 연결 · 배열 펼침"] {
        query_id
        context_position
        unit_id
    }
    class UNITS["법령 · units"] {
        unit_id
        index
        hierarchy
        content
        source_row_ids
    }
    class SPLITS["분할 · splits"] {
        query_id
        group_id
        split
    }
    QUERIES "1" --> "0..*" EVIDENCE : query_id
    QUERIES "1" --> "1" SPLITS : query_id
    EVIDENCE "1" --> "0..*" EVIDENCE_UNIT : query_id + context_position
    UNITS "1" --> "0..*" EVIDENCE_UNIT : unit_id
```

질문 하나에 여러 근거가 연결되고, 근거 하나는 여러 법령 항목으로 구성될 수 있습니다. 질문별 사용 구분은 `splits`에 저장합니다.

`EVIDENCE_UNIT`은 `unit_ids` 배열을 펼친 모습이며 별도 저장 파일은 아닙니다.
