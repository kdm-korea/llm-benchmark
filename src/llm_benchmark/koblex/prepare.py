"""원본 → 매핑 → 분할의 중간 객체를 연결하고 검증 후 저장한다."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
import json
from pathlib import Path
from typing import cast

from llm_benchmark.common.io import write_json, write_jsonl

from .audit import build_summary, validate_evaluation_data, verify_source_files
from .config import Settings, load_settings
from .mapping import build_qrels, build_units, map_queries
from .models import (
    AuditSummary, EvidenceMapping, MappedDataset, PathInput, PreparedDataset, Query,
    QuerySplit, QueryStatus, RawQuestion, RawStatute, SourceDataset, Unit,
)
from .splitting import split_queries


def load_sources(raw: Path, settings: Settings) -> SourceDataset:
    """파일 검증과 파싱을 끝내 계산 단계에 필요한 원본 객체를 반환한다."""
    import pyarrow.parquet as pq

    verify_source_files(raw, settings)
    tables = {
        'qa': pq.read_table(raw / settings.qa.file),
        'corpus': pq.read_table(raw / settings.corpus.file),
    }
    questions = cast(list[RawQuestion], tables['qa'].to_pylist())
    statutes = cast(list[RawStatute], tables['corpus'].to_pylist())
    question_ids = [question['id'] for question in questions]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError('Duplicate QA identifiers')

    return SourceDataset(
        questions=questions,
        statutes=statutes,
        schemas={
            name: [
                {'name': field.name, 'type': str(field.type), 'nullable': field.nullable}
                for field in table.schema
            ]
            for name, table in tables.items()
        },
    )


def map_sources(sources: SourceDataset, settings: Settings) -> MappedDataset:
    """파일 접근 없이 원본을 법령·질문·근거 연결·정답표로 변환한다."""
    units, units_by_index = build_units(sources.statutes, settings.corpus.revision)
    queries, mappings = map_queries(
        sources.questions, units_by_index, settings.qa.revision, settings.source_split,
    )
    ready_query_ids = {
        query.query_id for query in queries if query.evaluation_status == QueryStatus.READY
    }
    return MappedDataset(
        units=units,
        queries=queries,
        mappings=mappings,
        qrels=build_qrels(mappings, ready_query_ids),
    )


def partition_dataset(mapped: MappedDataset, settings: Settings) -> PreparedDataset:
    """매핑 결과는 유지하고 질문별 분할과 그룹 연결 정보를 추가한다."""
    splits, edges = split_queries(mapped.queries, settings.split, settings.source_split)
    return PreparedDataset(mapped=mapped, splits=splits, similarity_edges=edges)


def _write_dataset(output: Path, data: PreparedDataset) -> None:
    mapped = data.mapped
    records: dict[str, Sequence[Unit | Query | EvidenceMapping | QuerySplit]] = {
        'units': mapped.units,
        'queries': mapped.queries,
        'evidence': mapped.mappings,
        'splits': data.splits,
        'mapping-issues': [mapping for mapping in mapped.mappings if not mapping.unit_ids],
    }
    for name, rows in records.items():
        write_jsonl(output / f'{name}.jsonl', (asdict(record) for record in rows))
    write_json(output / 'qrels.json', mapped.qrels)


def _write_audit(audit_dir: Path, data: PreparedDataset) -> None:
    # 원문·후보 전체는 로컬에만 두고 공유 장부에는 연결과 판정만 남긴다.
    ledger = [
        {
            'query_id': mapping.query_id,
            'evidence_id': mapping.evidence_id,
            'context_position': mapping.context_position,
            'status': mapping.status,
            'unit_ids': mapping.unit_ids,
            'reason': mapping.reason,
            'duplicate_annotation': mapping.duplicate_annotation,
        }
        for mapping in data.mapped.mappings
    ]
    write_json(audit_dir / 'mapping-ledger.json', ledger)
    write_json(audit_dir / 'splits.json', [asdict(split) for split in data.splits])
    write_json(audit_dir / 'similarity-edges.json', data.similarity_edges)


def _save_dataset(
    data: PreparedDataset, sources: SourceDataset, settings: Settings,
    *, raw: Path, output: Path, audit_dir: Path,
) -> AuditSummary:
    _write_dataset(output, data)
    _write_audit(audit_dir, data)
    summary = build_summary(settings, raw, sources, data, output)
    summary['validation'] = {
        'all_source_hashes_verified': True,
        'qrel_units_exist': True,
        'ideal_metrics_all_one': True,
        'empty_metrics_all_zero': True,
        'no_group_crosses_dev_test': True,
    }
    write_json(output / 'manifest.json', summary)
    write_json(audit_dir / 'audit-summary.json', summary)
    return summary


def prepare(
    raw: PathInput | None = None, output: PathInput | None = None,
    audit_dir: PathInput | None = None, *, settings: Settings | None = None,
) -> None:
    settings = settings or load_settings()
    raw = Path(raw) if raw is not None else settings.raw
    output = Path(output) if output is not None else settings.output
    audit_dir = Path(audit_dir) if audit_dir is not None else settings.audit

    sources = load_sources(raw, settings)
    mapped = map_sources(sources, settings)
    dataset = partition_dataset(mapped, settings)
    validate_evaluation_data(dataset)
    summary = _save_dataset(
        dataset, sources, settings, raw=raw, output=output, audit_dir=audit_dir,
    )
    display_summary = {
        'qa_rows': summary['qa_rows'],
        'corpus_rows': summary['corpus_rows'],
        'mapping_status': summary['mapping_status'],
        'evaluation_status': summary['evaluation_status'],
        'split_counts': summary['split_counts'],
    }
    print(json.dumps(display_summary, ensure_ascii=False))
