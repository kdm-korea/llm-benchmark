"""Exact cosine retrieval, macro Recall@K and native MTEB verification."""

import importlib.metadata
import math
import os
from pathlib import Path
import warnings

import numpy as np


def exact_search(corpus_ids, corpus_vectors, query_ids, query_vectors, top_k=100, batch_size=32):
    """Compare each query with every document; break tied scores by descending ID."""
    corpus = np.array(corpus_vectors, dtype=np.float32, copy=True)
    queries = np.array(query_vectors, dtype=np.float32, copy=True)
    if corpus.ndim != 2 or queries.ndim != 2 or corpus.shape[1] != queries.shape[1]:
        raise ValueError("Incompatible vector shapes")
    if len(corpus_ids) != len(corpus) or len(query_ids) != len(queries):
        raise ValueError("Vector/ID counts differ")
    if not len(corpus) or top_k < 1 or batch_size < 1:
        raise ValueError("Invalid exact search size")
    if len(set(corpus_ids)) != len(corpus_ids) or len(set(query_ids)) != len(query_ids):
        raise ValueError("Duplicate search IDs")
    for vectors in (corpus, queries):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(vectors).all() or not np.isfinite(norms).all() or (norms == 0).any():
            raise ValueError("Non-finite or zero embedding")
        vectors /= norms  # Arrays below are isolated copies from cache.
    # Descending document IDs match MTEB/pytrec_eval's score tie handling.
    order = np.argsort(np.asarray(corpus_ids), kind="stable")[::-1]
    corpus = corpus[order]
    ids = [corpus_ids[i] for i in order]
    result = {}
    for start in range(0, len(queries), batch_size):
        scores = queries[start:start + batch_size] @ corpus.T
        for qid, score in zip(query_ids[start:start + batch_size], scores):
            best = np.argsort(-score, kind="stable")[:min(top_k, len(ids))]
            result[qid] = [(ids[i], float(score[i])) for i in best]
    return result


def evaluate_recall(queries, qrels, rankings, ks):
    """Return macro Recall, per-query results and explicit exclusion reasons."""
    if not ks or any(type(k) is not int or k <= 0 for k in ks):
        raise ValueError("K values must be positive integers")
    rows, excluded = [], []
    for qid, query in queries.items():
        relevant = {d for d, grade in qrels[qid].items() if grade > 0}
        if not relevant:
            excluded.append({"query_id": qid, "reason": "no_positive_qrels"})
            continue
        ranked = rankings[qid]
        docids = [d for d, _ in ranked]
        if len(docids) != len(set(docids)):
            raise ValueError(f"Duplicate retrieved document: {qid}")
        row = {
            "query_id": qid,
            "query_text": query,
            "relevant_doc_ids": sorted(relevant),
            "retrieved_doc_ids_top100": docids,
            "scores": [float(score) for _, score in ranked],
        }
        for k in ks:
            row[f"recall_at_{k}"] = len(set(docids[:k]) & relevant) / len(relevant)
        rows.append(row)
    if not rows:
        raise ValueError("No evaluable queries: all queries have no positive qrels")
    summary = {
        f"recall_at_{k}": sum(row[f"recall_at_{k}"] for row in rows) / len(rows)
        for k in ks
    }
    return summary, rows, excluded


def evaluate_with_mteb(qrels, rankings, ks, expected):
    """Verify Recall using the same metric function as MTEB's evaluator.

    Pin the private API here. No Task object or second search is necessary when
    document rankings are already available.
    """
    if importlib.metadata.version("mteb") != "2.26.11":
        raise RuntimeError("MTEB verification requires mteb==2.26.11")
    os.environ.setdefault("MTEB_CACHE", str(Path("cache/mteb").resolve()))
    from mteb._evaluators.retrieval_metrics import calculate_retrieval_scores

    positive_qrels = {
        query_id: {doc_id: 1 for doc_id, grade in grades.items() if grade > 0}
        for query_id, grades in qrels.items()
        if any(grade > 0 for grade in grades.values())
    }
    scores = {query_id: dict(rankings[query_id]) for query_id in positive_qrels}

    # Auxiliary nAUC may be undefined for a small inference/test dataset.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        native = calculate_retrieval_scores(scores, positive_qrels, ks)

    for k in ks:
        if not math.isclose(
            native.recall[f"Recall@{k}"],
            expected[f"recall_at_{k}"],
            abs_tol=0.000006,  # MTEB rounds aggregated Recall to five decimals.
        ):
            raise ValueError(f"MTEB/local Recall mismatch at K={k}")

    return _json_safe(native._asdict())


def _json_safe(value):
    """Keep native fields and serialize undefined auxiliary metrics as null."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, "_asdict"):
        return _json_safe(value._asdict())
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
