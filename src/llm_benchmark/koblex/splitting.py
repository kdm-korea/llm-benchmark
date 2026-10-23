"""유사한 질문을 같은 그룹에 묶고 개발·최종 평가용으로 분할한다."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import itertools
import re

from llm_benchmark.common.hashing import compact_digest

from .config import SplitSettings
from .models import GroupAssignment, Query, QuerySplit, SimilarityEdge, SplitName, QueryStatus


def _shingles(text: str, size: int) -> set[str]:
    text = re.sub(r'\s+', '', text)
    return {text[start:start + size] for start in range(max(1, len(text) - size + 1))}


def _jaccard(left: set[str], right: set[str]) -> float:
    combined = left | right
    return len(left & right) / len(combined) if combined else 1


def _group_queries(
    queries: Sequence[Query], settings: SplitSettings,
) -> tuple[list[list[Query]], list[SimilarityEdge]]:
    parent = list(range(len(queries)))

    def find_root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    query_shingles = [
        (
            _shingles(query.scenario, settings.shingle_size),
            _shingles(query.retrieval_query, settings.shingle_size),
        )
        for query in queries
    ]
    edges: list[SimilarityEdge] = []
    for left_index, right_index in itertools.combinations(range(len(queries)), 2):
        left = queries[left_index]
        right = queries[right_index]
        left_family = re.match(r'^qa_(\d+)_', left.query_id)
        right_family = re.match(r'^qa_(\d+)_', right.query_id)
        same_family = bool(left_family and right_family and left_family[1] == right_family[1])
        scenario_similarity = _jaccard(query_shingles[left_index][0], query_shingles[right_index][0])
        query_similarity = _jaccard(query_shingles[left_index][1], query_shingles[right_index][1])
        similar_text = max(scenario_similarity, query_similarity) >= settings.similarity_threshold
        if not same_family and not similar_text:
            continue

        parent[find_root(right_index)] = find_root(left_index)
        edges.append({
            'a': left.query_id,
            'b': right.query_id,
            'source_id_family': same_family,
            'scenario_jaccard': scenario_similarity,
            'query_jaccard': query_similarity,
        })

    groups: defaultdict[int, list[Query]] = defaultdict(list)
    for index, query in enumerate(queries):
        groups[find_root(index)].append(query)
    return list(groups.values()), edges


def split_queries(
    queries: Sequence[Query], settings: SplitSettings, source_split: str,
) -> tuple[list[QuerySplit], list[SimilarityEdge]]:
    """그룹을 유지하며 설정한 개발 비율에 가장 가까운 분할을 만든다."""
    groups, edges = _group_queries(queries, settings)

    def group_order(group: list[Query]) -> str:
        query_ids = sorted(query.query_id for query in group)
        return compact_digest([settings.seed, query_ids])

    ordered_groups = sorted(groups, key=group_order)
    ready_count = sum(query.evaluation_status == QueryStatus.READY for query in queries)
    target_dev_count = round(ready_count * settings.dev_ratio)
    dev_count = 0
    splits: list[QuerySplit] = []
    for group in ordered_groups:
        group_ready_count = sum(query.evaluation_status == QueryStatus.READY for query in group)
        current_distance = abs(dev_count - target_dev_count)
        distance_with_group = abs(dev_count + group_ready_count - target_dev_count)
        assign_to_dev = group_ready_count > 0 and distance_with_group < current_distance
        if assign_to_dev:
            dev_count += group_ready_count

        group_assignment: GroupAssignment = SplitName.DEV if assign_to_dev else SplitName.TEST
        query_ids = sorted(query.query_id for query in group)
        group_id = 'g_' + compact_digest(query_ids)[:16]
        for query in group:
            split = (
                group_assignment
                if query.evaluation_status == QueryStatus.READY
                else SplitName.HOLDOUT
            )
            splits.append(QuerySplit(
                query_id=query.query_id,
                group_id=group_id,
                split=split,
                group_assignment=group_assignment,
                source_split=source_split,
            ))
    return sorted(splits, key=lambda row: row.query_id), edges
