"""전체 문서 코사인 정확 검색, 질의별 Recall 평균 계산 및 MTEB 결과 대조."""

import importlib.metadata
import math
import os
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .models import Query, QueryEvaluation, Rankings, RecallEvaluation
from .parsing import mapping


def exact_search(
    corpus_ids: Sequence[str],
    corpus_vectors: ArrayLike,
    query_ids: Sequence[str],
    query_vectors: ArrayLike,
    top_k: int = 100,
    batch_size: int = 32,
) -> Rankings:
    """질의 벡터를 전체 문서 벡터와 비교해 코사인 유사도 상위 결과를 반환한다.

    동점은 문서 ID 내림차순으로 정렬해 MTEB와 같은 순위 규칙을 사용한다."""
    corpus = np.array(corpus_vectors, dtype=np.float32, copy=True)
    queries = np.array(query_vectors, dtype=np.float32, copy=True)
    if corpus.ndim != 2 or queries.ndim != 2 or corpus.shape[1] != queries.shape[1]:
        raise ValueError('Incompatible vector shapes')
    if len(corpus_ids) != len(corpus) or len(query_ids) != len(queries):
        raise ValueError('Vector/ID counts differ')
    if not len(corpus) or top_k < 1 or batch_size < 1:
        raise ValueError('Invalid exact search size')
    if len(set(corpus_ids)) != len(corpus_ids) or len(set(query_ids)) != len(query_ids):
        raise ValueError('Duplicate search IDs')
    for vectors in (corpus, queries):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(vectors).all() or not np.isfinite(norms).all() or (norms == 0).any():
            raise ValueError('Non-finite or zero embedding')
        vectors /= norms  # 캐시와 분리한 복사본만 정규화하므로 원본 벡터는 변경하지 않는다.
    # 동점의 문서 ID를 내림차순으로 정렬해 MTEB/pytrec_eval과 검색 순위를 맞춘다.
    order = np.argsort(np.asarray(corpus_ids), kind='stable')[::-1]
    corpus = corpus[order]
    ids = [corpus_ids[i] for i in order]
    result: Rankings = {}
    for start in range(0, len(queries), batch_size):
        # L2 정규화한 벡터의 내적은 코사인 유사도이므로 행렬 곱으로 일괄 계산한다.
        scores = queries[start : start + batch_size] @ corpus.T
        for qid, score in zip(query_ids[start : start + batch_size], scores):
            best = np.argsort(-score, kind='stable')[: min(top_k, len(ids))]
            result[qid] = [(ids[i], float(score[i])) for i in best]
    return result


def evaluate_recall(
    queries: Mapping[str, Query],
    qrels: Mapping[str, Mapping[str, float]],
    rankings: Rankings,
    ks: Sequence[int],
) -> RecallEvaluation:
    """질의별 관련 문서 회수율을 구해 같은 가중치로 평균낸다.

    질의별 순위·점수와 정답이 없어 평가에서 제외한 질의의 사유도 함께 반환한다."""
    if not ks or any(type(k) is not int or k <= 0 for k in ks):
        raise ValueError('K values must be positive integers')
    rows: list[QueryEvaluation] = []
    excluded: list[dict[str, str]] = []
    for qid, query in queries.items():
        relevant = {d for d, grade in qrels[qid].items() if grade > 0}
        if not relevant:
            excluded.append({'query_id': qid, 'reason': 'no_positive_qrels'})
            continue
        ranked = rankings[qid]
        docids = [d for d, _ in ranked]
        if len(docids) != len(set(docids)):
            raise ValueError(f'Duplicate retrieved document: {qid}')
        row = QueryEvaluation(
            query_id=qid,
            query_text=query.text,
            relevant_doc_ids=sorted(relevant),
            retrieved_doc_ids=docids,
            scores=[float(score) for _, score in ranked],
            recall={f'recall_at_{k}': len(set(docids[:k]) & relevant) / len(relevant) for k in ks},
        )
        rows.append(row)
    if not rows:
        raise ValueError('No evaluable queries: all queries have no positive qrels')
    summary = {
        f'recall_at_{k}': sum(row.recall[f'recall_at_{k}'] for row in rows) / len(rows) for k in ks
    }
    return RecallEvaluation(summary=summary, queries=rows, excluded=excluded)


def evaluate_with_mteb(
    qrels: Mapping[str, Mapping[str, float]],
    rankings: Rankings,
    ks: list[int],
    expected: Mapping[str, float],
) -> dict[str, object]:
    """직접 계산한 Recall을 MTEB 평가기가 사용하는 지표 함수의 결과와 대조한다.

    비공개 API 의존은 이 함수에 모으고 MTEB 버전을 고정한다. 이미 구한 검색
    순위를 재사용하므로 별도 Task 생성이나 중복 검색은 필요하지 않다.
    """
    if importlib.metadata.version('mteb') != '2.26.11':
        raise RuntimeError('MTEB verification requires mteb==2.26.11')
    os.environ.setdefault('MTEB_CACHE', str(Path('cache/mteb').resolve()))
    from mteb._evaluators.retrieval_metrics import calculate_retrieval_scores

    positive_qrels = {
        query_id: {doc_id: 1 for doc_id, grade in grades.items() if grade > 0}
        for query_id, grades in qrels.items()
        if any(grade > 0 for grade in grades.values())
    }
    scores = {query_id: dict(rankings[query_id]) for query_id in positive_qrels}

    # 소규모 검증 데이터에서는 보조 지표 nAUC가 정의되지 않을 수 있다.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        native = calculate_retrieval_scores(scores, positive_qrels, ks)

    for k in ks:
        if not math.isclose(
            native.recall[f'Recall@{k}'],
            expected[f'recall_at_{k}'],
            abs_tol=0.000006,  # MTEB가 평균 Recall을 소수점 다섯 자리로 반올림하는 오차를 허용한다.
        ):
            raise ValueError(f'MTEB/local Recall mismatch at K={k}')

    return mapping(_json_safe(native._asdict()))


def _json_safe(value: object) -> object:
    """MTEB 결과 필드는 유지하고, 정의되지 않은 보조 지표는 JSON의 null로 저장한다."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, '_asdict'):
        return _json_safe(value._asdict())
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, 'item'):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
