"""데이터 준비 → 임베딩 → 검색 → 평가 → 저장 순서로 실험을 실행한다.

CLI 인자 처리는 __main__에 두고 모델별 실행과 결과 저장을 연결한다.
YAML 설정의 상대 경로는 명령을 실행한 작업 디렉터리를 기준으로 해석한다.
"""

import csv
import importlib.metadata
import platform
import sys
import time
import uuid
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from llm_benchmark.common.io import write_json

from .dataset import RetrievalData, prepare_dataset
from .embeddings import EmbeddingModel, cached_encode, resolve_model
from .evaluation import evaluate_recall, evaluate_with_mteb, exact_search
from .models import (
    BenchmarkConfig,
    EncodedDataset,
    InputKind,
    ModelConfig,
    ModelEvaluation,
    ModelProgress,
    RunStatus,
    RuntimeConfig,
)


def select_models(config: BenchmarkConfig, names: Sequence[str] | None = None) -> list[ModelConfig]:
    """설정에 있는 모델 순서를 유지하고, 존재하지 않는 모델 이름은 오류로 처리한다."""
    models = config['models']
    if not names:
        return models

    unknown = set(names) - {model['name'] for model in models}
    if unknown:
        raise ValueError(f'Unknown models: {sorted(unknown)}')
    return [model for model in models if model['name'] in names]


def _new_run_id() -> str:
    timestamp = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%dT%H%M%S%z')
    return f'{timestamp}-{uuid.uuid4().hex[:8]}'


def _create_manifest(
    config: BenchmarkConfig,
    data: RetrievalData,
    validation: dict[str, object],
    query_limit: int | None,
    run_id: str,
) -> dict[str, object]:
    packages = ('numpy', 'torch', 'transformers', 'sentence-transformers', 'mteb')
    return {
        'run_id': run_id,
        'started_at': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        'status': RunStatus.RUNNING,
        'config': config,
        'mode': 'full' if len(data.queries) == validation['query_count'] else 'query_subset',
        'query_limit': query_limit,
        'corpus_count': len(data.corpus),
        'query_count': len(data.queries),
        'validation': validation,
        'platform': platform.platform(),
        'python': platform.python_version(),
        'working_directory': str(Path.cwd()),
        'versions': {name: importlib.metadata.version(name) for name in packages},
        'similarity': 'cosine',
        'tie_break': 'docid_descending',
        'system_prompt': None,
        'mteb_qrels': 'binarized_grade_gt_zero',
        'primary_metric': 'macro_recall',
        'models': {},
    }


def _encode_dataset(
    model_config: ModelConfig,
    runtime: RuntimeConfig,
    data: RetrievalData,
) -> EncodedDataset:
    """문서·질의 벡터를 만든 뒤 모델 메모리를 해제해 검색 단계의 메모리 부담을 줄인다."""
    model = EmbeddingModel(model_config, runtime)
    try:
        documents = cached_encode(
            model, list(data.corpus), data.document_texts(), InputKind.DOCUMENT, data.metadata
        )
        queries = cached_encode(
            model, list(data.queries), data.query_texts(), InputKind.QUERY, data.metadata
        )
    finally:
        model.close()
    return EncodedDataset(documents=documents, queries=queries)


def _evaluate_dataset(
    data: RetrievalData,
    encoded: EncodedDataset,
    config: BenchmarkConfig,
) -> ModelEvaluation:
    """검색 순위 하나를 자체 Recall과 MTEB 검증에 함께 사용한다."""
    ks = config['evaluation']['k']
    rankings = exact_search(
        list(data.corpus),
        encoded.documents.vectors,
        list(data.queries),
        encoded.queries.vectors,
        top_k=max(100, max(ks)),
        batch_size=config['evaluation']['query_batch_size'],
    )
    recall = evaluate_recall(data.queries, data.qrels, rankings, ks)
    native = evaluate_with_mteb(data.qrels, rankings, ks, recall.summary)
    return ModelEvaluation(recall=recall, native=native)


def _save_model_results(
    output_dir: Path,
    model: ModelConfig,
    data: RetrievalData,
    result: ModelEvaluation,
) -> None:
    rows = [
        {
            **row.to_dict(),
            'model_id': model['id'],
            'model_revision': model['revision'],
            'dataset_revision': data.metadata.revision,
        }
        for row in result.recall.queries
    ]
    write_json(output_dir / 'raw' / f'{model["name"]}.json', result.native)
    write_json(
        output_dir / 'per_query' / f'{model["name"]}.json',
        {'queries': rows, 'excluded': result.recall.excluded},
    )


def _save_manifest(
    path: Path,
    manifest: dict[str, object],
    progress: dict[str, ModelProgress],
) -> None:
    write_json(
        path, {**manifest, 'models': {name: item.to_dict() for name, item in progress.items()}}
    )


def _write_summary(path: Path, summaries: list[dict[str, object]], ks: Sequence[int]) -> None:
    fields = [
        'model',
        'model_id',
        'model_revision',
        'evaluated_queries',
        'excluded_queries',
        *[f'recall_at_{k}' for k in ks],
    ]
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)


def run_benchmark(
    config: BenchmarkConfig,
    selected_models: Sequence[str] | None = None,
    query_limit: int | None = None,
) -> Path:
    """선택한 모델을 전체 코퍼스에서 평가하고 모델 하나가 끝날 때마다 결과를 저장한다.

    뒤의 모델이 실패해도 앞서 완료한 모델의 결과와 실행 상태를 확인할 수 있게 한다."""
    models = select_models(config, selected_models)
    data, validation = prepare_dataset(config)
    limit = query_limit if query_limit is not None else config['evaluation']['query_limit']
    data = data.subset_queries(limit)

    run_id = _new_run_id()
    output_dir = Path(config['runtime']['results_dir']) / run_id
    output_dir.mkdir(parents=True)
    manifest = _create_manifest(config, data, validation, limit, run_id)
    manifest_path = output_dir / 'manifest.json'
    progress: dict[str, ModelProgress] = {}
    _save_manifest(manifest_path, manifest, progress)
    summaries: list[dict[str, object]] = []
    ks = config['evaluation']['k']
    print(f'Run: {output_dir}', flush=True)

    try:
        for configured_model in models:
            started = time.monotonic()
            model = resolve_model(configured_model)
            model_status = ModelProgress(resolved_config=model)
            progress[model['name']] = model_status
            _save_manifest(manifest_path, manifest, progress)
            print(f'Encoding {model["name"]} ({model["revision"]})', flush=True)

            encoded = _encode_dataset(model, config['runtime'], data)
            result = _evaluate_dataset(data, encoded, config)
            _save_model_results(output_dir, model, data, result)
            recall = result.recall
            summaries.append(
                {
                    'model': model['name'],
                    'model_id': model['id'],
                    'model_revision': model['revision'],
                    'evaluated_queries': len(recall.queries),
                    'excluded_queries': len(recall.excluded),
                    **recall.summary,
                }
            )
            _write_summary(output_dir / 'recall_summary.csv', summaries, ks)
            model_status.status = RunStatus.COMPLETE
            model_status.details.update(
                document_stats=encoded.documents.stats,
                query_stats=encoded.queries.stats,
                evaluated_queries=len(recall.queries),
                excluded_queries=recall.excluded,
                elapsed_seconds=time.monotonic() - started,
            )
            _save_manifest(manifest_path, manifest, progress)
            print(model['name'], recall.summary, flush=True)
        manifest['status'] = RunStatus.COMPLETE
    except BaseException as error:
        status = RunStatus.INTERRUPTED if isinstance(error, KeyboardInterrupt) else RunStatus.FAILED
        manifest.update(status=status, error=f'{type(error).__name__}: {error}')
        for model_status in progress.values():
            if model_status.status == RunStatus.RUNNING:
                model_status.status = status
                model_status.details['error'] = manifest['error']
        raise
    finally:
        manifest['finished_at'] = datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
        _save_manifest(manifest_path, manifest, progress)
    return output_dir


def probe_models(config: BenchmarkConfig, selected_models: Sequence[str] | None = None) -> Path:
    """문서 2개·질의 2개로 실제 모델 추론이 가능한지 확인한다.

    검색 성능 실험이 아니므로 Recall 점수는 산출하지 않는다."""
    models = select_models(config, selected_models)
    data, _ = prepare_dataset(config)
    output_path = Path(config['runtime']['results_dir']) / 'probes' / f'{_new_run_id()}.json'
    model_reports: dict[str, object] = {}
    report: dict[str, object] = {
        'kind': 'inference_probe',
        'benchmark_score': False,
        'status': RunStatus.RUNNING,
        'models': model_reports,
        'dataset_revision': data.metadata.revision,
        'config': config,
    }
    write_json(output_path, report)
    try:
        for configured_model in models:
            started = time.monotonic()
            model_config = resolve_model(configured_model)
            model = EmbeddingModel(model_config, config['runtime'])
            try:
                documents = model.encode(data.document_texts()[:2], InputKind.DOCUMENT)
                queries = model.encode(data.query_texts()[:2], InputKind.QUERY)
            finally:
                model.close()
            model_reports[model_config['name']] = {
                'resolved_config': model_config,
                'document_shape': list(documents.vectors.shape),
                'query_shape': list(queries.vectors.shape),
                'document_stats': documents.stats,
                'query_stats': queries.stats,
                'elapsed_seconds': time.monotonic() - started,
            }
            write_json(output_path, report)
        report['status'] = RunStatus.COMPLETE
    except BaseException as error:
        report.update(status=RunStatus.FAILED, error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output_path, report)
    return output_path


def main() -> int:
    """기존 run 모듈의 직접 실행은 새 CLI 진입점으로 위임한다."""
    from .__main__ import main as cli_main

    return cli_main()


if __name__ == '__main__':
    sys.exit(main())
