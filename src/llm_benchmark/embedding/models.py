"""외부 모델 라이브러리 없이 읽을 수 있는 실험 데이터와 단계별 결과 타입."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, NotRequired, TypedDict

from llm_benchmark.embedding.parsing import string

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class DuplicateQueryPolicy(StrEnum):
    ERROR = "error"
    MERGE_MAX = "merge_max"


class InputKind(StrEnum):
    DOCUMENT = "document"
    QUERY = "query"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class CorpusRow(TypedDict):
    docid: str
    text: str
    title: NotRequired[str]
    kind: NotRequired[str]


class QrelsRow(TypedDict):
    query: str
    grades: dict[str, float]


class ModelConfig(TypedDict):
    name: str
    id: str
    revision: str
    max_length: int
    query_prefix: str
    document_prefix: str


class DatasetConfig(TypedDict):
    id: str
    revision: str
    raw_dir: str
    prepared_dir: str
    split: str
    input_template: str
    duplicate_query_policy: str


class RuntimeConfig(TypedDict):
    device: str
    batch_size: int
    cache_dir: str
    model_cache_dir: str
    results_dir: str


class EvaluationConfig(TypedDict):
    k: list[int]
    query_batch_size: int
    query_limit: int | None


class BenchmarkConfig(TypedDict):
    dataset: DatasetConfig
    runtime: RuntimeConfig
    evaluation: EvaluationConfig
    models: list[ModelConfig]


@dataclass(frozen=True)
class Document:
    title: str
    text: str
    kind: str = ""

    def to_dict(self) -> dict[str, str]:
        """기존 준비 데이터와 fingerprint에 쓰던 필드 순서를 유지한다."""
        return {"title": self.title, "text": self.text, "kind": self.kind}

    def embedding_text(self) -> str:
        return "\n".join(part for part in (self.title.strip(), self.text.strip()) if part)


@dataclass(frozen=True)
class Query:
    text: str


@dataclass
class PreparedQueries:
    queries: dict[str, Query]
    qrels: dict[str, dict[str, float]]
    duplicate_rows: list[dict[str, object]]
    source_rows: int


@dataclass
class DataMetadata:
    source: dict[str, object]
    source_qrels_rows: int
    duplicate_query_policy: DuplicateQueryPolicy
    actual_file_sha256: dict[str, str]
    duplicate_rows: list[dict[str, object]]
    data_hash: str = ""

    @property
    def revision(self) -> str:
        return string(self.source["revision"])

    def to_dict(self) -> dict[str, object]:
        return {
            **self.source,
            "split": "test",
            "query_id_policy": "sha256-json-text-first24",
            "source_qrels_rows": self.source_qrels_rows,
            "duplicate_query_policy": self.duplicate_query_policy.value,
            "actual_file_sha256": self.actual_file_sha256,
            "duplicate_rows": self.duplicate_rows,
            "data_hash": self.data_hash,
        }


Rankings = dict[str, list[tuple[str, float]]]


@dataclass
class QueryEvaluation:
    query_id: str
    query_text: str
    relevant_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    scores: list[float]
    recall: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "relevant_doc_ids": self.relevant_doc_ids,
            "retrieved_doc_ids_top100": self.retrieved_doc_ids,
            "scores": self.scores,
            **self.recall,
        }


@dataclass
class RecallEvaluation:
    summary: dict[str, float]
    queries: list[QueryEvaluation]
    excluded: list[dict[str, str]]


@dataclass
class EncodedBatch:
    vectors: NDArray[np.float32]
    stats: dict[str, object]


@dataclass
class EncodedDataset:
    documents: EncodedBatch
    queries: EncodedBatch


@dataclass
class ModelEvaluation:
    recall: RecallEvaluation
    native: dict[str, object]


@dataclass
class ModelProgress:
    resolved_config: ModelConfig
    status: RunStatus = RunStatus.RUNNING
    details: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {"resolved_config": self.resolved_config, "status": self.status.value, **self.details}
