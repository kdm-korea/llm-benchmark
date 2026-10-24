"""CLI 분리 후 새 진입점과 기존 진입점이 같은 준비 결과를 만드는지 검증한다."""

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize('module', ['llm_benchmark.embedding', 'llm_benchmark.embedding.run'])
def test_cli_prepare_preserves_outputs(
    module: str,
    tmp_path: Path,
    make_source: Callable[..., None],
) -> None:
    root = Path(__file__).resolve().parents[2]
    raw = tmp_path / 'raw'
    raw.mkdir()
    make_source(raw, [{'query': 'question', 'grades': {'001': 2}}])
    config = yaml.safe_load((root / 'configs/legal_recall.yaml').read_text())
    config['dataset'].update(
        id='test',
        revision='abc',
        raw_dir=str(raw),
        prepared_dir=str(tmp_path / 'prepared'),
    )
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(json.dumps(config))

    result = subprocess.run(
        [sys.executable, '-m', module, 'prepare', '--config', str(config_path)],
        cwd=root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['corpus_count'] == 2
    assert report['query_count'] == 1
    output = tmp_path / 'prepared'
    assert json.loads((output / 'corpus.json').read_text())['001']['text'] == 'body'
    assert list(json.loads((output / 'qrels.json').read_text()).values()) == [{'001': 2}]
