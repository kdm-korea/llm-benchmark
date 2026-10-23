"""임베딩 테스트 안에서만 쓰는 fixture. 다른 테스트 파일을 import해 재사용하지 않는다."""

import json

import pytest


def _make_source(tmp_path, qrels, corpus=None):
    if corpus is None:
        corpus = [
            {"docid": "001", "text": "body", "title": "title"},
            {"docid": "002", "text": "other"},
        ]
    source = {"dataset_id": "test", "revision": "abc"}
    (tmp_path / "source.json").write_text(json.dumps(source))
    for name, rows in [("corpus", corpus), ("qrels", qrels)]:
        content = ''.join(json.dumps(row) + '\n' for row in rows)
        (tmp_path / f"{name}.jsonl").write_text(content)


@pytest.fixture
def make_source():
    return _make_source


@pytest.fixture(autouse=True)
def isolate_mteb_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MTEB_CACHE", str(tmp_path / "mteb"))
