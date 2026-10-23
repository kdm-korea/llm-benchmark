"""Offline checks of the actual experiment flow and native MTEB integration."""

import json

import numpy as np
import pytest

import embedding_benchmark.run as benchmark
from test_dataset import make_source


@pytest.fixture
def experiment_config(tmp_path):
    config = benchmark.load_config('configs/legal_recall.yaml')
    raw = tmp_path / 'raw'
    raw.mkdir()
    make_source(raw, [
        {'query': 'first', 'grades': {'001': 2}},
        {'query': 'second', 'grades': {'002': 1}},
    ])
    config['dataset'].update(
        id='test', revision='abc', raw_dir=str(raw),
        prepared_dir=str(tmp_path / 'prepared'),
    )
    config['runtime']['results_dir'] = str(tmp_path / 'results')
    config['models'] = config['models'][:1]
    return config


def test_run_writes_coherent_outputs_without_model_network(experiment_config, monkeypatch):
    class FakeModel:
        def __init__(self, *args):
            pass

        def close(self):
            pass

    def encode_fixture(*args):
        return np.eye(2, dtype=np.float32), {'cache_hit': False}

    monkeypatch.setattr(benchmark, 'resolve_model', lambda model: model)
    monkeypatch.setattr(benchmark, 'EmbeddingModel', FakeModel)
    monkeypatch.setattr(benchmark, 'cached_encode', encode_fixture)

    output = benchmark.run_benchmark(experiment_config)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['mode'] == 'full'
    assert manifest['corpus_count'] == 2

    native = json.loads((output / 'raw/bge-m3.json').read_text())
    assert native['recall']['Recall@100'] == 1.
    per_query = json.loads((output / 'per_query/bge-m3.json').read_text())
    assert len(per_query['queries']) == 2
    assert (output / 'recall_summary.csv').exists()


def test_failure_manifest_is_preserved(experiment_config, tmp_path, monkeypatch):
    def unavailable_model(model):
        raise ValueError('model not available')

    monkeypatch.setattr(benchmark, 'resolve_model', unavailable_model)
    with pytest.raises(ValueError, match='model not available'):
        benchmark.run_benchmark(experiment_config)

    output = next((tmp_path / 'results').iterdir())
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'failed'
    assert 'model not available' in manifest['error']


def test_probe_reports_inference_only_and_releases_model(experiment_config, monkeypatch):
    closed_models = []

    class FakeModel:
        def __init__(self, config, runtime):
            self.name = config['name']

        def encode(self, texts, kind):
            return np.ones((len(texts), 3)), {'count': len(texts)}

        def close(self):
            closed_models.append(self.name)

    monkeypatch.setattr(benchmark, 'resolve_model', lambda model: model)
    monkeypatch.setattr(benchmark, 'EmbeddingModel', FakeModel)

    output = benchmark.probe_models(experiment_config, ['bge-m3'])
    report = json.loads(output.read_text())
    assert report['status'] == 'complete'
    assert report['benchmark_score'] is False
    assert report['models']['bge-m3']['document_shape'] == [2, 3]
    assert report['models']['bge-m3']['query_shape'] == [2, 3]
    assert closed_models == ['bge-m3']
