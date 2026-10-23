import numpy as np
import pytest

from llm_benchmark.embedding.evaluation import evaluate_recall, evaluate_with_mteb, exact_search

def test_exhaustive_cosine_normalizes_without_mutating_and_breaks_ties_like_mteb():
    corpus = np.array([[2., 0.], [1., 0.], [0., 1.]], dtype=np.float32)
    query = np.array([[3., 0.]], dtype=np.float32)
    before = corpus.copy()
    corpus.flags.writeable = False
    result = exact_search(["a", "b", "c"], corpus, ["q"], query, top_k=100)
    assert [d for d, _ in result["q"]] == ["b", "a", "c"]
    np.testing.assert_array_equal(corpus, before)
    np.testing.assert_array_equal(query, [[3., 0.]])


def test_query_batching_preserves_rankings():
    rng = np.random.default_rng(42)
    corpus, queries = rng.normal(size=(23, 8)), rng.normal(size=(7, 8))
    args = (list(map(str, range(23))), corpus, list(map(str, range(7))), queries)
    first = exact_search(*args, top_k=5, batch_size=1)
    second = exact_search(*args, top_k=5, batch_size=7)
    assert {q: [d for d, _ in hits] for q, hits in first.items()} == {q: [d for d, _ in hits] for q, hits in second.items()}


@pytest.mark.parametrize('bad', [np.array([[0., 0.]]), np.array([[float('nan'), 1.]])])
def test_invalid_vectors_fail(bad):
    with pytest.raises(ValueError, match="embedding"):
        exact_search(["d"], bad, ["q"], np.array([[1., 0.]]))


def test_macro_recall_uses_all_positives_and_excludes_no_positive_query():
    queries = {"a": "first", "b": "second", "c": "excluded"}
    qrels = {"a": {"d1": 2, "d2": 1}, "b": {"d3": 1}, "c": {"d1": 0}}
    rankings = {"a": [("d1", 1), ("d3", .5)], "b": [("d3", 1)], "c": []}
    scores, rows, excluded = evaluate_recall(queries, qrels, rankings, [1, 5])
    assert scores == {"recall_at_1": .75, "recall_at_5": .75}
    assert len(rows) == 2
    assert excluded == [{"query_id": "c", "reason": "no_positive_qrels"}]


def test_all_excluded_is_not_a_zero_score():
    with pytest.raises(ValueError, match="No evaluable"):
        evaluate_recall({"q": "query"}, {"q": {}}, {"q": []}, [1])


def test_duplicate_retrieved_ids_fail():
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_recall({"q": "query"}, {"q": {"d": 1}}, {"q": [("d", 1), ("d", .5)]}, [1])


def test_native_mteb_recall_matches_macro_with_ties_and_nonpositive_exclusion():
    queries = {"q1": "first", "q2": "second", "q3": "exclude"}
    qrels = {"q1": {"a": 2, "b": 1}, "q2": {"b": 1}, "q3": {}}
    rankings = {"q1": [("b", 1.), ("a", 1.)], "q2": [("a", 1.), ("b", .5)], "q3": []}
    expected, _, _ = evaluate_recall(queries, qrels, rankings, [1, 5, 10, 20, 50, 100])
    result = evaluate_with_mteb(qrels, rankings, [1, 5, 10, 20, 50, 100], expected)
    assert result["recall"]["Recall@1"] == pytest.approx(.25)
    assert result["recall"]["Recall@100"] == 1.