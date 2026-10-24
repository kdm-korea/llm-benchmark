"""모델 추론을 테스트 대역으로 바꿔 실행 흐름과 실제 MTEB 연동을 네트워크 없이 검증한다."""

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest

import llm_benchmark.embedding.run as benchmark
from llm_benchmark.embedding.config import load_config
from llm_benchmark.embedding.models import (
    BenchmarkConfig,
    EncodedBatch,
    InputKind,
    ModelConfig,
    RuntimeConfig,
)


@pytest.fixture
def experiment_config(tmp_path: Path, make_source: Callable[..., None]) -> BenchmarkConfig:
    config = load_config(Path(__file__).resolve().parents[2] / 'configs/legal_recall.yaml')
    raw = tmp_path / 'raw'
    raw.mkdir()
    make_source(
        raw,
        [
            {'query': 'first', 'grades': {'001': 2}},
            {'query': 'second', 'grades': {'002': 1}},
        ],
    )
    config['dataset'].update(
        id='test',
        revision='abc',
        raw_dir=str(raw),
        prepared_dir=str(tmp_path / 'prepared'),
    )
    config['runtime']['results_dir'] = str(tmp_path / 'results')
    config['models'] = config['models'][:1]
    return config


def test_run_writes_coherent_outputs_without_model_network(
    experiment_config: BenchmarkConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        def __init__(self, *args: object) -> None:
            pass

        def close(self) -> None:
            pass

    def encode_fixture(*args: object) -> EncodedBatch:
        return EncodedBatch(vectors=np.eye(2, dtype=np.float32), stats={'cache_hit': False})

    monkeypatch.setattr(benchmark, 'resolve_model', lambda model: model)
    monkeypatch.setattr(benchmark, 'EmbeddingModel', FakeModel)
    monkeypatch.setattr(benchmark, 'cached_encode', encode_fixture)

    output = benchmark.run_benchmark(experiment_config)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['mode'] == 'full'
    assert manifest['corpus_count'] == 2

    native = json.loads((output / 'raw/bge-m3.json').read_text())
    assert native['recall']['Recall@100'] == 1.0
    per_query = json.loads((output / 'per_query/bge-m3.json').read_text())
    assert len(per_query['queries']) == 2
    assert (output / 'recall_summary.csv').exists()


def test_failure_manifest_is_preserved(
    experiment_config: BenchmarkConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_model(model: ModelConfig) -> ModelConfig:
        raise ValueError('model not available')

    monkeypatch.setattr(benchmark, 'resolve_model', unavailable_model)
    with pytest.raises(ValueError, match='model not available'):
        benchmark.run_benchmark(experiment_config)

    output = next((tmp_path / 'results').iterdir())
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'failed'
    assert 'model not available' in manifest['error']


def test_probe_reports_inference_only_and_releases_model(
    experiment_config: BenchmarkConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed_models = []

    class FakeModel:
        def __init__(self, config: ModelConfig, runtime: RuntimeConfig) -> None:
            self.name = config['name']

        def encode(self, texts: Sequence[str], kind: InputKind) -> EncodedBatch:
            return EncodedBatch(vectors=np.ones((len(texts), 3)), stats={'count': len(texts)})

        def close(self) -> None:
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


@pytest.mark.parametrize('failure', [ValueError('encoding failed'), KeyboardInterrupt()])
def test_encoding_failure_releases_model_and_records_active_status(
    experiment_config: BenchmarkConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: BaseException,
) -> None:
    closed = []

    class FailingModel:
        def __init__(self, config: ModelConfig, runtime: RuntimeConfig) -> None:
            self.name = config['name']

        def close(self) -> None:
            closed.append(self.name)

    def fail_encoding(*args: object) -> EncodedBatch:
        raise failure

    monkeypatch.setattr(benchmark, 'resolve_model', lambda model: model)
    monkeypatch.setattr(benchmark, 'EmbeddingModel', FailingModel)
    monkeypatch.setattr(benchmark, 'cached_encode', fail_encoding)
    with pytest.raises(type(failure)):
        benchmark.run_benchmark(experiment_config)
    manifest = json.loads(next((tmp_path / 'results').glob('*/manifest.json')).read_text())
    expected = 'interrupted' if isinstance(failure, KeyboardInterrupt) else 'failed'
    assert manifest['status'] == expected
    assert manifest['models']['bge-m3']['status'] == expected
    assert 'finished_at' in manifest
    assert closed == ['bge-m3']


def test_probe_failure_releases_model(
    experiment_config: BenchmarkConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    closed = []

    class FailingModel:
        def __init__(self, config: ModelConfig, runtime: RuntimeConfig) -> None:
            pass

        def encode(self, texts: Sequence[str], kind: InputKind) -> EncodedBatch:
            raise ValueError('probe failed')

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(benchmark, 'resolve_model', lambda model: model)
    monkeypatch.setattr(benchmark, 'EmbeddingModel', FailingModel)
    with pytest.raises(ValueError, match='probe failed'):
        benchmark.probe_models(experiment_config)
    report = json.loads(next((tmp_path / 'results/probes').glob('*.json')).read_text())
    assert report['status'] == 'failed'
    assert closed == [True]


def test_later_model_failure_keeps_completed_outputs(
    experiment_config: BenchmarkConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first_model = experiment_config['models'][0]
    experiment_config['models'].append({**first_model, 'name': 'second-model'})
    closed = []

    class FakeModel:
        def __init__(self, config: ModelConfig, runtime: RuntimeConfig) -> None:
            self.name = config['name']

        def close(self) -> None:
            closed.append(self.name)

    def encode_fixture(model: FakeModel, *args: object) -> EncodedBatch:
        if model.name == 'second-model':
            raise ValueError('second model failed')
        return EncodedBatch(vectors=np.eye(2, dtype=np.float32), stats={'cache_hit': False})

    monkeypatch.setattr(benchmark, 'resolve_model', lambda model: model)
    monkeypatch.setattr(benchmark, 'EmbeddingModel', FakeModel)
    monkeypatch.setattr(benchmark, 'cached_encode', encode_fixture)
    with pytest.raises(ValueError, match='second model failed'):
        benchmark.run_benchmark(experiment_config)
    output = next((tmp_path / 'results').iterdir())
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'failed'
    assert manifest['models']['bge-m3']['status'] == 'complete'
    assert manifest['models']['second-model']['status'] == 'failed'
    assert (output / 'per_query/bge-m3.json').exists()
    assert not (output / 'per_query/second-model.json').exists()
    assert len((output / 'recall_summary.csv').read_text().splitlines()) == 2
    assert closed == ['bge-m3', 'second-model']
