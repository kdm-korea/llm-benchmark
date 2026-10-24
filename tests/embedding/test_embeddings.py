import shutil
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import llm_benchmark.embedding.embeddings as embeddings
from llm_benchmark.embedding.embeddings import cached_encode
from llm_benchmark.embedding.models import (
    DataMetadata,
    DuplicateQueryPolicy,
    EncodedBatch,
    InputKind,
)


def _dataset_metadata() -> DataMetadata:
    return DataMetadata(
        source={'revision': 'dataset'},
        source_qrels_rows=0,
        duplicate_query_policy=DuplicateQueryPolicy.ERROR,
        actual_file_sha256={},
        duplicate_rows=[],
    )


class FakeEncoder:
    def __init__(self, path: str | Path) -> None:
        self.runtime = {'cache_dir': str(path)}
        self.config = {'id': 'fixture', 'revision': 'abc'}
        self.device = 'cpu'
        self.model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(dtype='float32')]))
        self.calls = 0

    def encode(self, texts: Sequence[str], kind: InputKind) -> EncodedBatch:
        self.calls += 1
        return EncodedBatch(
            vectors=np.ones((len(texts), 3), dtype=np.float32), stats={'count': len(texts)}
        )


def test_cache_reuses_identical_inputs_but_invalidates_text_and_id_order(tmp_path: Path) -> None:
    encoder = FakeEncoder(tmp_path)
    args = (
        encoder,
        ['a', 'b'],
        ['first', 'second'],
        InputKind.DOCUMENT,
        _dataset_metadata(),
    )
    first = cached_encode(*args).stats
    second = cached_encode(*args)
    assert not first['cache_hit'] and second.stats['cache_hit'] and encoder.calls == 1
    assert not second.vectors.flags.writeable
    changed = cached_encode(
        encoder,
        ['b', 'a'],
        ['second', 'first'],
        InputKind.DOCUMENT,
        _dataset_metadata(),
    ).stats
    assert not changed['cache_hit'] and encoder.calls == 2
    assert first['cache_key'] != changed['cache_key']


def test_corrupted_cache_is_rejected(tmp_path: Path) -> None:
    encoder = FakeEncoder(tmp_path)
    args = (encoder, ['a'], ['first'], InputKind.QUERY, _dataset_metadata())
    metadata = cached_encode(*args).stats
    path = tmp_path / (metadata['cache_key'] + '.npy')
    path.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        cached_encode(*args)


def test_reads_cache_metadata_from_before_dataclass_refactor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """이전 구현에서 만든 고정 키와 메타데이터로 새 인코딩 없이 읽을 수 있어야 한다."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(embeddings.importlib.metadata, 'version', lambda name: 'fixture-version')
    fixture = (
        Path(__file__).parent
        / 'fixtures/a55fca5c4bb56b7f5dfd1524d4173a1dcc1936f933a866195c926fde88f9d3f9.json'
    )
    cache = tmp_path / 'vectors'
    cache.mkdir()
    shutil.copyfile(fixture, cache / fixture.name)
    np.save(cache / f'{fixture.stem}.npy', np.ones((2, 3), dtype=np.float32), allow_pickle=False)
    encoder = FakeEncoder('vectors')
    result = cached_encode(
        encoder,
        ['001', '002'],
        ['title\nbody', 'other'],
        InputKind.DOCUMENT,
        _dataset_metadata(),
    )
    assert result.stats == {'count': 2, 'cache_hit': True, 'cache_key': fixture.stem}
    assert encoder.calls == 0
