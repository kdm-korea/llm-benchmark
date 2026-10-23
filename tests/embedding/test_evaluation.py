import numpy as np
import pytest

from llm_benchmark.embedding.models import Query

from llm_benchmark.embedding.evaluation import evaluate_recall, evaluate_with_mteb, exact_search

def test_exhaustive_cosine_normalizes_without_mutating_and_breaks_ties_like_mteb() -> None:
    corpus = np.array([[2., 0.], [1., 0.], [0., 1.]], dtype=np.float32)
    query = np.array([[3., 0.]], dtype=np.float32)
    before = corpus.copy()
    corpus.flags.writeable = False
    result = exact_search(["a", "b", "c"], corpus, ["q"], query, top_k=100)
    assert [d for d, _ in result["q"]] == ["b", "a", "c"]
    np.testing.assert_array_equal(corpus, before)
    np.testing.assert_array_equal(query, [[3., 0.]])


def test_query_batching_preserves_rankings() -> None:
    rng = np.random.default_rng(42)
    corpus, queries = rng.normal(size=(23, 8)), rng.normal(size=(7, 8))
    args = (list(map(str, range(23))), corpus, list(map(str, range(7))), queries)
    first = exact_search(*args, top_k=5, batch_size=1)
    second = exact_search(*args, top_k=5, batch_size=7)
    assert {q: [d for d, _ in hits] for q, hits in first.items()} == {q: [d for d, _ in hits] for q, hits in second.items()}


@pytest.mark.parametrize('bad', [np.array([[0., 0.]]), np.array([[float('nan'), 1.]])])
def test_invalid_vectors_fail(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="embedding"):
        exact_search(["d"], bad, ["q"], np.array([[1., 0.]]))


def test_macro_recall_uses_all_positives_and_excludes_no_positive_query() -> None:
    queries = {"a": Query("first"), "b": Query("second"), "c": Query("excluded")}
    qrels = {"a": {"d1": 2, "d2": 1}, "b": {"d3": 1}, "c": {"d1": 0}}
    rankings = {"a": [("d1", 1), ("d3", .5)], "b": [("d3", 1)], "c": []}
    result = evaluate_recall(queries, qrels, rankings, [1, 5])
    assert result.summary == {"recall_at_1": .75, "recall_at_5": .75}
    assert len(result.queries) == 2
    assert result.excluded == [{"query_id": "c", "reason": "no_positive_qrels"}]


def test_all_excluded_is_not_a_zero_score() -> None:
    with pytest.raises(ValueError, match="No evaluable"):
        evaluate_recall({"q": Query("query")}, {"q": {}}, {"q": []}, [1])


def test_duplicate_retrieved_ids_fail() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_recall({"q": Query("query")}, {"q": {"d": 1}}, {"q": [("d", 1), ("d", .5)]}, [1])


def test_native_mteb_recall_matches_macro_with_ties_and_nonpositive_exclusion() -> None:
    queries = {"q1": Query("first"), "q2": Query("second"), "q3": Query("exclude")}
    qrels = {"q1": {"a": 2, "b": 1}, "q2": {"b": 1}, "q3": {}}
    rankings = {"q1": [("b", 1.), ("a", 1.)], "q2": [("a", 1.), ("b", .5)], "q3": []}
    evaluation = evaluate_recall(queries, qrels, rankings, [1, 5, 10, 20, 50, 100])
    result = evaluate_with_mteb(qrels, rankings, [1, 5, 10, 20, 50, 100], evaluation.summary)
    assert result["recall"]["Recall@1"] == pytest.approx(.25)
    assert result["recall"]["Recall@100"] == 1.

def test_native_ndcg_binarizes_all_positive_grades() -> None:
    rankings = {"q": [("wrong", 1.0), ("a", 0.8), ("b", 0.5)]}
    queries = {"q": Query("query")}
    ks = [1, 5, 10, 20, 50, 100]
    binary = {"q": {"a": 1, "b": 1}}
    graded = {"q": {"a": 8, "b": 2, "wrong": 0}}
    expected = evaluate_recall(queries, binary, rankings, ks).summary
    assert evaluate_with_mteb(binary, rankings, ks, expected) == evaluate_with_mteb(
        graded, rankings, ks, expected,
    )
