"""내부 처리 객체와 원본·출력 JSON 스키마를 정의한다."""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal, NotRequired, Protocol, TypeAlias, TypedDict


PathInput: TypeAlias = str | Path


class MappingStatus(StrEnum):
    EXACT = 'exact'
    ASSEMBLED = 'assembled'
    ASSEMBLED_REORDERED = 'assembled_reordered'
    AMBIGUOUS = 'ambiguous'
    UNRESOLVED = 'unresolved'


class UnitKind(StrEnum):
    STRUCTURAL_HEADING = 'structural_heading'
    ARTICLE_TITLE = 'article_title'
    ARTICLE_BODY = 'article_body'
    PARAGRAPH = 'paragraph'
    ITEM = 'item'
    OTHER = 'other'


class ParentStatus(StrEnum):
    UNRESOLVED = 'unresolved'
    STRUCTURAL_UNRESOLVED = 'structural_unresolved'
    ARTICLE_OR_UNRESOLVED_ROOT = 'article_or_unresolved_root'
    UNIQUE_SYNTACTIC = 'unique_syntactic'
    AMBIGUOUS = 'ambiguous'


class QueryStatus(StrEnum):
    READY = 'ready'
    HOLD = 'hold'


class ReviewStatus(StrEnum):
    AUTOMATED_MAPPING = 'automated_mapping'
    AUTOMATED_EXACTNESS_CHECK = 'automated_exactness_check'


class SplitName(StrEnum):
    DEV = 'dev'
    TEST = 'test'
    HOLDOUT = 'holdout'


GroupAssignment: TypeAlias = Literal[SplitName.DEV, SplitName.TEST]


class RawStatute(TypedDict):
    index: str
    hierarchy: str
    content: str


class EvidenceContext(RawStatute):
    index_eng: NotRequired[str]
    hierarchy_eng: NotRequired[str]
    content_eng: NotRequired[str]


class RawQuestion(TypedDict):
    id: str
    background: str
    question: str
    answer: str
    contexts: list[EvidenceContext]
    n_hops: int
    background_eng: NotRequired[str]
    question_eng: NotRequired[str]
    answer_eng: NotRequired[str]


@dataclass(slots=True, kw_only=True)
class Unit:
    index: str
    hierarchy: str
    content: str
    unit_id: str
    source_row_ids: list[int] = field(default_factory=list)
    law_name: str | None = None
    article_label: str | None = None
    kind: UnitKind
    parent_id: str | None = None
    parent_candidates: list[str] = field(default_factory=list)
    parent_status: ParentStatus = ParentStatus.UNRESOLVED
    effective_date: None = None
    source_revision: str


@dataclass(slots=True, kw_only=True)
class Query:
    query_id: str
    source_row_id: int
    source_split: str
    scenario: str
    question: str
    retrieval_query: str
    reference_answer: str
    n_hops: int
    source_revision: str
    case_date: None = None
    evaluation_status: QueryStatus = QueryStatus.READY
    review_status: ReviewStatus = ReviewStatus.AUTOMATED_MAPPING


@dataclass(slots=True, kw_only=True)
class EvidenceMapping:
    query_id: str
    evidence_id: str
    context_position: int
    original_context: EvidenceContext
    status: MappingStatus
    unit_ids: list[str]
    reason: str
    duplicate_annotation: bool = False
    candidate_unit_ids: list[str] = field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.AUTOMATED_EXACTNESS_CHECK


@dataclass(slots=True, frozen=True, kw_only=True)
class QuerySplit:
    query_id: str
    group_id: str
    split: SplitName
    group_assignment: GroupAssignment
    source_split: str


@dataclass(slots=True, frozen=True, kw_only=True)
class MatchResult:
    status: MappingStatus
    unit_ids: list[str]
    reason: str


class SimilarityEdge(TypedDict):
    a: str
    b: str
    source_id_family: bool
    scenario_jaccard: float
    query_jaccard: float


# 한 근거에 연결된 법령 ID는 모두 회수해야 하므로 목록을 유지한다.
Qrels: TypeAlias = dict[str, dict[str, list[str]]]
UnitIndex: TypeAlias = dict[str, list[Unit]]
Rankings: TypeAlias = dict[str, list[set[str]]]
MetricRow: TypeAlias = dict[str, str | float]


class CoverageUnit(Protocol):
    @property
    def content(self) -> str: ...

    @property
    def kind(self) -> UnitKind: ...


class SourceSpan(TypedDict):
    unit_id: str
    start: int
    end: int
    text: str


class SourceFile(TypedDict):
    path: str
    url: str
    size_bytes: int
    sha256: str


class ManifestSource(TypedDict):
    repo: str
    revision: str
    files: list[SourceFile]


class SourceManifest(TypedDict):
    raw_root: str
    retrieved_at_utc: NotRequired[str]
    sources: list[ManifestSource]


class SourceSelection(TypedDict):
    repo: str
    file: str
    duplicate_file: str


class QuestionSelection(SourceSelection):
    split: str


class SplitParameters(TypedDict):
    seed: int
    dev_ratio: float
    similarity_threshold: float
    shingle_size: int


class PathsConfig(TypedDict):
    raw: str
    output: str
    audit: str


class DownloadConfig(TypedDict):
    timeout_seconds: float


class ExperimentConfig(TypedDict):
    normalization_version: str
    source_manifest: str
    paths: PathsConfig
    qa: QuestionSelection
    corpus: SourceSelection
    split: SplitParameters
    download: DownloadConfig


class FileMetadata(TypedDict):
    sha256: str
    size_bytes: int


class FieldSchema(TypedDict):
    name: str
    type: str
    nullable: bool


class RuntimeInfo(TypedDict):
    python: str
    pyarrow: str


class RunConfiguration(TypedDict):
    qa_file: str
    corpus_file: str
    source_split: str
    split: SplitParameters


class ValidationResult(TypedDict):
    all_source_hashes_verified: bool
    qrel_units_exist: bool
    ideal_metrics_all_one: bool
    empty_metrics_all_zero: bool
    no_group_crosses_dev_test: bool


class AuditSummary(TypedDict):
    normalization_version: str
    source_manifest_sha256: str
    runtime: RuntimeInfo
    configuration: RunConfiguration
    qa_rows: int
    corpus_rows: int
    canonical_units: int
    exact_duplicate_rows: int
    index_unique: int
    index_duplicated_values: int
    hierarchy_unique: int
    hierarchy_duplicated_values: int
    hierarchies_with_different_bodies: int
    schema: dict[str, list[FieldSchema]]
    null_or_empty: dict[str, dict[str, int]]
    context_null_or_empty: dict[str, int]
    train_file_equivalence: dict[str, bool]
    unit_kinds: dict[str, int]
    raw_row_kinds: dict[str, int]
    parent_status: dict[str, int]
    n_hops: dict[int, int]
    context_counts: dict[int, int]
    n_hops_context_count_mismatches: list[str]
    normalized_unique: dict[str, int]
    mapping_status: dict[str, int]
    duplicate_evidence_annotations: int
    evaluation_status: dict[str, int]
    held_queries: list[str]
    split_counts: dict[str, int]
    groups: int
    similarity_edges: int
    shared_dev_test_evidence: int
    artifacts: dict[str, FileMetadata]
    validation: NotRequired[ValidationResult]


@dataclass(slots=True, frozen=True, kw_only=True)
class SourceDataset:
    """원본 읽기 단계의 결과. 파싱된 행과 원본 스키마를 함께 보존한다."""

    questions: list[RawQuestion]
    statutes: list[RawStatute]
    schemas: dict[str, list[FieldSchema]]


@dataclass(slots=True, frozen=True, kw_only=True)
class MappedDataset:
    """원본 질문의 근거가 내부 법령 ID에 연결된 결과."""

    units: list[Unit]
    queries: list[Query]
    mappings: list[EvidenceMapping]
    qrels: Qrels


@dataclass(slots=True, frozen=True, kw_only=True)
class PreparedDataset:
    """매핑 결과와 개발·최종 평가 분할을 묶은 저장 전 데이터."""

    mapped: MappedDataset
    splits: list[QuerySplit]
    similarity_edges: list[SimilarityEdge]
