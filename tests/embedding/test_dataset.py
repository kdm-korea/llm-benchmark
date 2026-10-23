from collections.abc import Callable
from pathlib import Path
import pytest
from llm_benchmark.embedding.dataset import load_eve


def test_duplicate_policy_is_explicit_and_auditable(tmp_path: Path, make_source: Callable[..., None]) -> None:
    make_source(tmp_path, [{"query": "same", "grades": {"001": 2}},
                           {"query": "same", "grades": {"001": 1, "002": 2}}])
    with pytest.raises(ValueError, match="Duplicate"):
        load_eve(tmp_path)
    data = load_eve(tmp_path, "merge_max")
    assert len(data.queries) == 1
    assert next(iter(data.qrels.values())) == {"001": 2, "002": 2}
    assert len(data.metadata.duplicate_rows) == 1
    assert data.document_texts() == ["title\nbody", "other"]


def test_unknown_document_fails(tmp_path: Path, make_source: Callable[..., None]) -> None:
    make_source(tmp_path, [{"query": "q", "grades": {"missing": 1}}])
    with pytest.raises(ValueError, match="Unknown"):
        load_eve(tmp_path)


def test_duplicate_corpus_fails(tmp_path: Path, make_source: Callable[..., None]) -> None:
    make_source(tmp_path, [{"query": "q", "grades": {"001": 1}}],
                [{"docid": "001", "text": "a"}, {"docid": "001", "text": "b"}])
    with pytest.raises(ValueError, match="Duplicate corpus"):
        load_eve(tmp_path)


def test_query_subset_keeps_entire_corpus(tmp_path: Path, make_source: Callable[..., None]) -> None:
    make_source(tmp_path, [{"query": "first", "grades": {"001": 1}}, {"query": "second", "grades": {"002": 1}}])
    data = load_eve(tmp_path)
    subset = data.subset_queries(1)
    assert len(subset.queries) == 1 and subset.corpus == data.corpus


@pytest.mark.parametrize('grade', [True, -1, float('nan'), float('inf'), '1'])
def test_invalid_duplicate_grade_cannot_be_hidden(tmp_path: Path, make_source: Callable[..., None], grade: object) -> None:
    make_source(tmp_path, [
        {"query": "same", "grades": {"001": 2}},
        {"query": "same", "grades": {"001": grade}},
    ])
    with pytest.raises(ValueError, match="Invalid relevance grade"):
        load_eve(tmp_path, "merge_max")
