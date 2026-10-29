"""Evidence retrieval metrics and source character span validation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import cast

from .models import CoverageUnit, MetricRow, Qrels, Rankings, SourceSpan, UnitKind


def evaluate_evidence(
    qrels: Qrels, rankings: Rankings, ks: Sequence[int] = (1, 5, 10, 20, 50, 100),
) -> tuple[dict[str, float], list[MetricRow]]:
    """qrels[qid][evidence_id] = all required canonical units for that evidence.

    rankings[qid] is an ordered list of sets of fully included canonical units.
    A split unit contributes nothing until the caller proves its full body was
    reconstructed. Each chunk consumes rank even when it repeats earlier units.
    """
    if not ks or any(type(k) is not int or k <= 0 for k in ks):
        raise ValueError('K values must be positive integers')
    if set(qrels) != set(rankings):
        raise ValueError('Query IDs differ; missing rankings must be explicit empty lists')
    rows: list[MetricRow] = []
    for qid, evidence in qrels.items():
        if not evidence or any(not units for units in evidence.values()):
            raise ValueError(f'Unresolved or empty evidence: {qid}')
        covered: set[str] = set()
        first: int | None = None
        counts: list[int] = []
        for rank, chunk_units in enumerate(rankings[qid], 1):
            covered.update(chunk_units)
            count = sum(set(units) <= covered for units in evidence.values())
            if count and first is None:
                first = rank
            counts.append(count)
        row: MetricRow = {'query_id': qid, 'reciprocal_rank': 1 / first if first else 0.0}
        for k in ks:
            count = counts[min(k, len(counts)) - 1] if counts else 0
            row[f'evidence_recall_at_{k}'] = count / len(evidence)
            row[f'hit_at_{k}'] = int(count > 0)
            row[f'all_evidence_hit_at_{k}'] = int(count == len(evidence))
        rows.append(row)
    if not rows:
        raise ValueError('No evaluable queries')
    summary = {
        key: sum(cast(float, row[key]) for row in rows) / len(rows)
        for key in rows[0] if key != 'query_id'
    }
    summary['mrr'] = summary.pop('reciprocal_rank')
    return summary, rows


def chunk_coverage_rankings(
    units: Mapping[str, CoverageUnit], chunks: Sequence[Sequence[SourceSpan]],
) -> list[set[str]]:
    """Validate raw character spans and yield newly completed bodies per rank.

    Each chunk is a list of {unit_id, start, end, text} spans. Text must actually
    be present in the retrieved chunk; this function checks it against the raw
    canonical body, not just against a claimed evidence ID. Coordinates use Python
    Unicode character offsets, half open, in the immutable unit content.
    """
    intervals: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    completed: set[str] = set()
    ranking: list[set[str]] = []
    for chunk in chunks:
        newly_completed: set[str] = set()
        for span in chunk:
            uid, start, end = span['unit_id'], span['start'], span['end']
            unit = units[uid]
            body = unit.content
            if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(body)
                    or span['text'] != body[start:end]):
                raise ValueError(f'Invalid source span: {uid}')
            if unit.kind in (UnitKind.STRUCTURAL_HEADING, UnitKind.ARTICLE_TITLE):
                continue
            intervals[uid].append((start, end))
            cursor = 0
            for left, right in sorted(intervals[uid]):
                if left > cursor:
                    break
                cursor = max(cursor, right)
            if cursor == len(body) and uid not in completed:
                newly_completed.add(uid)
                completed.add(uid)
        ranking.append(newly_completed)
    return ranking
