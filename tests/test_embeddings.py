from types import SimpleNamespace
import numpy as np
import pytest
from embedding_benchmark.embeddings import cached_encode


class FakeEncoder:
    def __init__(self, path):
        self.runtime = {'cache_dir': str(path)}
        self.config = {'id': 'fixture', 'revision': 'abc'}
        self.device = 'cpu'
        self.model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(dtype='float32')]))
        self.calls = 0

    def encode(self, texts, kind):
        self.calls += 1
        return np.ones((len(texts), 3), dtype=np.float32), {'count': len(texts)}


def test_cache_reuses_identical_inputs_but_invalidates_text_and_id_order(tmp_path):
    encoder = FakeEncoder(tmp_path)
    args = (encoder, ['a', 'b'], ['first', 'second'], 'document', {'revision': 'dataset'})
    _, first = cached_encode(*args)
    vectors, second = cached_encode(*args)
    assert not first['cache_hit'] and second['cache_hit'] and encoder.calls == 1
    assert not vectors.flags.writeable
    _, changed = cached_encode(encoder, ['b', 'a'], ['second', 'first'], 'document', {'revision': 'dataset'})
    assert not changed['cache_hit'] and encoder.calls == 2
    assert first['cache_key'] != changed['cache_key']


def test_corrupted_cache_is_rejected(tmp_path):
    encoder = FakeEncoder(tmp_path)
    args = (encoder, ['a'], ['first'], 'query', {'revision': 'dataset'})
    _, metadata = cached_encode(*args)
    path = tmp_path / (metadata['cache_key'] + '.npy')
    path.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        cached_encode(*args)
