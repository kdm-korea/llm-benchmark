"""원본 근거를 법령 항목에 연결하고 질문별 정답 집합을 만든다."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import re

from llm_benchmark.common.hashing import compact_digest

from .models import (
    EvidenceContext, EvidenceMapping, MatchResult, Qrels, Query, RawQuestion,
    RawStatute, Unit, UnitIndex, UnitKind, MappingStatus, ParentStatus, QueryStatus,
)


def normalize(text: str) -> str:
    """공백만 정리하고 문장부호·숫자·개정 표시는 유지한다."""
    return ' '.join(text.split())


def row_kind(content: str) -> UnitKind:
    if re.fullmatch(r'제\d+[편장절관].*', content):
        return UnitKind.STRUCTURAL_HEADING
    if re.match(r'^제\d+조(?:의\d+)?\s*\([^)]*\)\s*$', content):
        return UnitKind.ARTICLE_TITLE
    if re.match(r'^제\d+조', content):
        return UnitKind.ARTICLE_BODY
    if re.match(r'^[①-⑳㉑-㉟]', content):
        return UnitKind.PARAGRAPH
    if re.match(r'^\d+(?:의\d+)?\.', content):
        return UnitKind.ITEM
    return UnitKind.OTHER


def build_units(rows: Sequence[RawStatute], revision: str) -> tuple[list[Unit], UnitIndex]:
    """원본 세 필드가 같은 행만 합치고 모든 출처 위치를 유지한다."""
    units_by_content: dict[tuple[str, str, str], Unit] = {}
    for source_position, row in enumerate(rows):
        content_key = (row['index'], row['hierarchy'], row['content'])
        if content_key not in units_by_content:
            units_by_content[content_key] = _create_unit(row, revision, content_key)
        units_by_content[content_key].source_row_ids.append(source_position)

    units = list(units_by_content.values())
    units_by_index: defaultdict[str, list[Unit]] = defaultdict(list)
    for unit in units:
        units_by_index[normalize(unit.index)].append(unit)
    for same_article_units in units_by_index.values():
        _assign_parents(same_article_units)
    return units, units_by_index


def _create_unit(
    row: RawStatute, revision: str, content_key: tuple[str, str, str],
) -> Unit:
    article_match = re.match(r'^(.*?)\s+(\d+조(?:\s+\d+)?)\b', row['index'])
    return Unit(
        index=row['index'],
        hierarchy=row['hierarchy'],
        content=row['content'],
        unit_id='u_' + compact_digest([revision, *content_key]),
        source_row_ids=[],
        law_name=article_match[1] if article_match else None,
        article_label=article_match[2] if article_match else None,
        kind=row_kind(row['content']),
        parent_id=None,
        parent_candidates=[],
        parent_status=ParentStatus.UNRESOLVED,
        effective_date=None,
        source_revision=revision,
    )


def _assign_parents(units: Sequence[Unit]) -> None:
    """같은 조문의 가장 긴 계층 접두사를 구문상 부모 후보로 기록한다."""
    units_by_hierarchy: defaultdict[str, list[Unit]] = defaultdict(list)
    for unit in units:
        if unit.kind != UnitKind.STRUCTURAL_HEADING:
            units_by_hierarchy[normalize(unit.hierarchy)].append(unit)

    for unit in units:
        if unit.kind == UnitKind.STRUCTURAL_HEADING:
            unit.parent_status = ParentStatus.STRUCTURAL_UNRESOLVED
            continue
        hierarchy = normalize(unit.hierarchy)
        parent_hierarchies = [
            parent for parent in units_by_hierarchy if hierarchy.startswith(parent + ' ')
        ]
        if not parent_hierarchies:
            unit.parent_status = ParentStatus.ARTICLE_OR_UNRESOLVED_ROOT
            continue

        nearest_parent = max(parent_hierarchies, key=len)
        parent_ids = sorted(parent.unit_id for parent in units_by_hierarchy[nearest_parent])
        unit.parent_candidates = parent_ids
        unit.parent_status = ParentStatus.UNIQUE_SYNTACTIC if len(parent_ids) == 1 else ParentStatus.AMBIGUOUS
        unit.parent_id = parent_ids[0] if len(parent_ids) == 1 else None


def match_context(
    context: EvidenceContext, units_by_index: Mapping[str, Sequence[Unit]],
) -> MatchResult:
    """단일 일치를 먼저 확인하고, 없을 때만 전체 본문 조합을 시도한다."""
    article_index = normalize(context['index'])
    hierarchy = normalize(context['hierarchy'])
    reference_body = normalize(context['content'])
    body_units = [
        unit for unit in units_by_index.get(article_index, [])
        if unit.kind not in (UnitKind.STRUCTURAL_HEADING, UnitKind.ARTICLE_TITLE)
    ]

    exact_matches = [
        unit for unit in body_units
        if normalize(unit.hierarchy) == hierarchy
        and normalize(unit.content) == reference_body
    ]
    if len(exact_matches) == 1:
        return MatchResult(
            status=MappingStatus.EXACT,
            unit_ids=[exact_matches[0].unit_id],
            reason='normalized index, hierarchy and full body agree',
        )
    if len(exact_matches) > 1:
        return MatchResult(
            status=MappingStatus.AMBIGUOUS,
            unit_ids=[],
            reason='multiple raw units agree after whitespace normalization',
        )

    return _match_assembled_body(body_units, hierarchy, reference_body)


def _match_assembled_body(
    body_units: Sequence[Unit], hierarchy: str, reference_body: str,
) -> MatchResult:
    candidates = [
        unit for unit in body_units
        if normalize(unit.hierarchy) == hierarchy
        or normalize(unit.hierarchy).startswith(hierarchy + ' ')
    ]
    hierarchy_counts = Counter(normalize(unit.hierarchy) for unit in candidates)
    has_root = hierarchy in hierarchy_counts
    unique_hierarchies = all(count == 1 for count in hierarchy_counts.values())
    if has_root and len(candidates) > 1 and unique_hierarchies:
        combined_body = ' '.join(normalize(unit.content) for unit in candidates)
        if combined_body == reference_body:
            return MatchResult(
                status=MappingStatus.ASSEMBLED,
                unit_ids=[unit.unit_id for unit in candidates],
                reason='ordered full bodies exactly reconstruct reference',
            )

        ordered_units = sorted(
            candidates, key=lambda unit: reference_body.find(normalize(unit.content)),
        )
        reordered_body = ' '.join(normalize(unit.content) for unit in ordered_units)
        if reordered_body == reference_body:
            return MatchResult(
                status=MappingStatus.ASSEMBLED_REORDERED,
                unit_ids=[unit.unit_id for unit in ordered_units],
                reason='complete bodies reconstruct reference in QA item order',
            )
    return MatchResult(
        status=MappingStatus.UNRESOLVED,
        unit_ids=[],
        reason='same-article candidates do not exactly reconstruct reference body',
    )


def _create_query(
    row: RawQuestion, source_row_id: int, revision: str, source_split: str,
) -> Query:
    query = Query(
        query_id=row['id'],
        source_row_id=source_row_id,
        source_split=source_split,
        scenario=row['background'],
        question=row['question'],
        retrieval_query=row['background'].strip() + '\n\n' + row['question'].strip(),
        reference_answer=row['answer'],
        n_hops=row['n_hops'],
        source_revision=revision,
        case_date=None,
        evaluation_status=QueryStatus.READY,
    )
    if not all(row.get(key) for key in ('background', 'question', 'answer', 'contexts')):
        query.evaluation_status = QueryStatus.HOLD
    return query


def map_queries(
    qa: Sequence[RawQuestion], units_by_index: Mapping[str, Sequence[Unit]],
    revision: str, source_split: str,
) -> tuple[list[Query], list[EvidenceMapping]]:
    """원본 주석을 보존하고 미해결 근거가 있는 질문을 보류한다."""
    queries: list[Query] = []
    mappings: list[EvidenceMapping] = []
    for source_row_id, row in enumerate(qa):
        query = _create_query(row, source_row_id, revision, source_split)
        seen_evidence_ids: set[str] = set()
        for position, context in enumerate(row['contexts']):
            mapping = _map_evidence(row['id'], position, context, units_by_index)
            evidence_id = mapping.evidence_id
            mapping.duplicate_annotation = evidence_id in seen_evidence_ids
            seen_evidence_ids.add(evidence_id)
            mappings.append(mapping)
            if not mapping.unit_ids:
                query.evaluation_status = QueryStatus.HOLD
        queries.append(query)
    return queries, mappings


def _map_evidence(
    query_id: str, position: int, context: EvidenceContext,
    units_by_index: Mapping[str, Sequence[Unit]],
) -> EvidenceMapping:
    evidence_key = [
        normalize(context['index']), normalize(context['hierarchy']), normalize(context['content']),
    ]
    match = match_context(context, units_by_index)
    candidates = units_by_index.get(normalize(context['index']), [])
    return EvidenceMapping(
        query_id=query_id,
        evidence_id='e_' + compact_digest(evidence_key),
        context_position=position,
        original_context=context,
        status=match.status,
        unit_ids=match.unit_ids,
        reason=match.reason,
        duplicate_annotation=False,
        candidate_unit_ids=[unit.unit_id for unit in candidates],
    )


def build_qrels(mappings: Sequence[EvidenceMapping], ready_query_ids: set[str]) -> Qrels:
    """모든 근거가 연결된 질문만 정답 집합에 포함한다."""
    qrels: defaultdict[str, dict[str, list[str]]] = defaultdict(dict)
    for mapping in mappings:
        if mapping.query_id in ready_query_ids:
            qrels[mapping.query_id][mapping.evidence_id] = mapping.unit_ids
    return qrels
