"""원본·평가 데이터의 무결성을 검증하고 실행 통계를 구성한다."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import importlib.metadata
from pathlib import Path
import platform
from typing import cast

from llm_benchmark.common.hashing import file_sha256

from .config import Settings
from .evaluation import evaluate_evidence
from .mapping import normalize, row_kind
from .models import (
    AuditSummary, EvidenceMapping, FileMetadata, PreparedDataset, QuerySplit,
    Rankings, SourceDataset, SplitName, SplitParameters, QueryStatus,
)


def verify_source_files(raw: Path, settings: Settings) -> None:
    """산출물 생성 전에 고정 원본의 크기·해시를 검증한다."""
    for source in settings.manifest['sources']:
        for entry in source['files']:
            # 실행 경로가 달라도 manifest에 기록한 원본 식별 정보는 유지한다.
            path = settings.source_path(raw, entry)
            wrong_size = path.stat().st_size != entry['size_bytes']
            wrong_hash = file_sha256(path) != entry['sha256']
            if wrong_size or wrong_hash:
                raise ValueError(f'Checksum mismatch: {path}')


def validate_evaluation_data(data: PreparedDataset) -> None:
    """정답 참조·지표 극값·그룹 분할 무결성을 검증한다."""
    unit_ids = {unit.unit_id for unit in data.mapped.units}
    ideal_rankings: Rankings = {}
    for query_id, evidence in data.mapped.qrels.items():
        required_units = {unit_id for ids in evidence.values() for unit_id in ids}
        assert required_units <= unit_ids
        ideal_rankings[query_id] = [required_units]

    ideal_scores, _ = evaluate_evidence(data.mapped.qrels, ideal_rankings)
    empty_scores, _ = evaluate_evidence(data.mapped.qrels, {qid: [] for qid in data.mapped.qrels})
    assert all(score == 1 for score in ideal_scores.values())
    assert all(score == 0 for score in empty_scores.values())
    assignments: defaultdict[str, set[str]] = defaultdict(set)
    for split in data.splits:
        assignments[split.group_id].add(split.group_assignment)
    assert all(len(group_assignments) == 1 for group_assignments in assignments.values())


def _empty_field_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    return {
        field: sum(row[field] is None or row[field] == '' or row[field] == [] for row in rows)
        for field in rows[0]
    }


def _shared_evidence_count(
    mappings: Sequence[EvidenceMapping], splits: Sequence[QuerySplit], ready_query_ids: set[str],
) -> int:
    split_by_query_id = {row.query_id: row.split for row in splits}
    splits_by_evidence_id: defaultdict[str, set[SplitName]] = defaultdict(set)
    for mapping in mappings:
        query_id = mapping.query_id
        if query_id in ready_query_ids:
            splits_by_evidence_id[mapping.evidence_id].add(split_by_query_id[query_id])
    return sum(
        used_splits == {SplitName.DEV, SplitName.TEST}
        for used_splits in splits_by_evidence_id.values()
    )


def _artifact_metadata(output: Path) -> dict[str, FileMetadata]:
    artifacts: dict[str, FileMetadata] = {}
    for path in sorted(output.glob('*')):
        if path.is_file() and path.name != 'manifest.json':
            artifacts[path.name] = {
                'sha256': file_sha256(path),
                'size_bytes': path.stat().st_size,
            }
    return artifacts


def build_summary(
    settings: Settings, raw: Path, sources: SourceDataset, data: PreparedDataset, output: Path,
) -> AuditSummary:
    """단계별 결과 객체에서 통계를 모으고 기존 보고서 형식을 유지한다."""
    qa_rows = sources.questions
    corpus_rows = sources.statutes
    units = data.mapped.units
    queries = data.mapped.queries
    mappings = data.mapped.mappings
    splits = data.splits
    ready_query_ids = {
        query.query_id for query in queries if query.evaluation_status == QueryStatus.READY
    }
    index_counts = Counter(row['index'] for row in corpus_rows)
    hierarchy_counts = Counter(row['hierarchy'] for row in corpus_rows)
    bodies_by_hierarchy: defaultdict[str, set[str]] = defaultdict(set)
    for row in corpus_rows:
        bodies_by_hierarchy[row['hierarchy']].add(row['content'])

    contexts: list[Mapping[str, object]] = [context for row in qa_rows for context in row['contexts']]
    context_empty_counts = {
        field: sum(not context[field] for context in contexts)
        for field in qa_rows[0]['contexts'][0]
    }
    mismatched_hop_ids = [
        row['id'] for row in qa_rows if row['n_hops'] != len(row['contexts'])
    ]
    return {
        'normalization_version': settings.version,
        'source_manifest_sha256': file_sha256(settings.manifest_path),
        'runtime': {
            'python': platform.python_version(),
            'pyarrow': importlib.metadata.version('pyarrow'),
        },
        'configuration': {
            'qa_file': str(settings.qa.file),
            'corpus_file': str(settings.corpus.file),
            'source_split': settings.source_split,
            'split': cast(SplitParameters, asdict(settings.split)),
        },
        'qa_rows': len(qa_rows),
        'corpus_rows': len(corpus_rows),
        'canonical_units': len(units),
        'exact_duplicate_rows': len(corpus_rows) - len(units),
        'index_unique': len(index_counts),
        'index_duplicated_values': sum(count > 1 for count in index_counts.values()),
        'hierarchy_unique': len(hierarchy_counts),
        'hierarchy_duplicated_values': sum(count > 1 for count in hierarchy_counts.values()),
        'hierarchies_with_different_bodies': sum(
            len(bodies) > 1 for bodies in bodies_by_hierarchy.values()
        ),
        'schema': sources.schemas,
        'null_or_empty': {
            'qa': _empty_field_counts(qa_rows),
            'corpus': _empty_field_counts(corpus_rows),
        },
        'context_null_or_empty': context_empty_counts,
        'train_file_equivalence': {
            'qa_byte_identical_to_test': (
                (raw / settings.qa.duplicate_file).read_bytes()
                == (raw / settings.qa.file).read_bytes()
            ),
            'statute_byte_identical_to_corpus': (
                (raw / settings.corpus.duplicate_file).read_bytes()
                == (raw / settings.corpus.file).read_bytes()
            ),
        },
        'unit_kinds': dict(Counter(unit.kind.value for unit in units)),
        'raw_row_kinds': dict(Counter(row_kind(row['content']).value for row in corpus_rows)),
        'parent_status': dict(Counter(unit.parent_status.value for unit in units)),
        'n_hops': dict(Counter(row['n_hops'] for row in qa_rows)),
        'context_counts': dict(Counter(len(row['contexts']) for row in qa_rows)),
        'n_hops_context_count_mismatches': mismatched_hop_ids,
        'normalized_unique': {
            field: len({
                normalize(cast(str, row[field]))
                for row in cast(list[Mapping[str, object]], qa_rows)
            })
            for field in ('id', 'background', 'question', 'answer')
        },
        'mapping_status': dict(Counter(mapping.status.value for mapping in mappings)),
        'duplicate_evidence_annotations': sum(
            mapping.duplicate_annotation for mapping in mappings
        ),
        'evaluation_status': dict(Counter(query.evaluation_status.value for query in queries)),
        'held_queries': [
            query.query_id for query in queries if query.evaluation_status != QueryStatus.READY
        ],
        'split_counts': dict(Counter(row.split.value for row in splits)),
        'groups': len({row.group_id for row in splits}),
        'similarity_edges': len(data.similarity_edges),
        'shared_dev_test_evidence': _shared_evidence_count(mappings, splits, ready_query_ids),
        'artifacts': _artifact_metadata(output),
    }
