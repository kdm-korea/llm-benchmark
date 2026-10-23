"""데이터 준비 → 임베딩 → 검색 → 평가 → 저장 순서로 실험을 실행한다.

CLI와 실행 흐름을 같은 파일에 두어 전체 과정을 따라 읽을 수 있게 한다.
YAML 설정의 상대 경로는 명령을 실행한 작업 디렉터리를 기준으로 해석한다.
"""

import argparse
import csv
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import uuid
from zoneinfo import ZoneInfo

import yaml

from llm_benchmark.common.io import write_json
from llm_benchmark.embedding.dataset import download_dataset, prepare_dataset
from llm_benchmark.embedding.embeddings import EmbeddingModel, cached_encode, resolve_model
from llm_benchmark.embedding.evaluation import evaluate_recall, evaluate_with_mteb, exact_search


def select_models(config, names=None):
    """설정에 있는 모델 순서를 유지하고, 존재하지 않는 모델 이름은 오류로 처리한다."""
    models = config["models"]
    if not names:
        return models

    unknown = set(names) - {model["name"] for model in models}
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    return [model for model in models if model["name"] in names]


def _new_run_id():
    timestamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%dT%H%M%S%z")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _create_manifest(config, data, validation, query_limit, run_id):
    packages = ("numpy", "torch", "transformers", "sentence-transformers", "mteb")
    return {
        "run_id": run_id,
        "started_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "status": "running",
        "config": config,
        "mode": "full" if len(data.queries) == validation["query_count"] else "query_subset",
        "query_limit": query_limit,
        "corpus_count": len(data.corpus),
        "query_count": len(data.queries),
        "validation": validation,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "working_directory": str(Path.cwd()),
        "versions": {name: importlib.metadata.version(name) for name in packages},
        "similarity": "cosine",
        "tie_break": "docid_descending",
        "system_prompt": None,
        "mteb_qrels": "binarized_grade_gt_zero",
        "primary_metric": "macro_recall",
        "models": {},
    }


def _encode_dataset(model_config, runtime, data):
    """문서·질의 벡터를 만든 뒤 모델 메모리를 해제해 검색 단계의 메모리 부담을 줄인다."""
    model = EmbeddingModel(model_config, runtime)
    try:
        document_vectors, document_stats = cached_encode(
            model, list(data.corpus), data.document_texts(), "document", data.metadata
        )
        query_vectors, query_stats = cached_encode(
            model, list(data.queries), list(data.queries.values()), "query", data.metadata
        )
    finally:
        model.close()
    return document_vectors, query_vectors, document_stats, query_stats


def _save_model_results(output_dir, model, data, native, rows, excluded):
    for row in rows:
        row.update(
            model_id=model["id"],
            model_revision=model["revision"],
            dataset_revision=data.metadata["revision"],
        )
    write_json(output_dir / "raw" / f"{model['name']}.json", native)
    write_json(
        output_dir / "per_query" / f"{model['name']}.json",
        {"queries": rows, "excluded": excluded},
    )


def _write_summary(path, summaries, ks):
    fields = [
        "model", "model_id", "model_revision", "evaluated_queries", "excluded_queries",
        *[f"recall_at_{k}" for k in ks],
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)


def run_benchmark(config, selected_models=None, query_limit=None):
    """선택한 모델을 전체 코퍼스에서 평가하고 모델 하나가 끝날 때마다 결과를 저장한다.

    뒤의 모델이 실패해도 앞서 완료한 모델의 결과와 실행 상태를 확인할 수 있게 한다."""
    models = select_models(config, selected_models)
    data, validation = prepare_dataset(config)
    limit = query_limit if query_limit is not None else config["evaluation"]["query_limit"]
    data = data.subset_queries(limit)

    run_id = _new_run_id()
    output_dir = Path(config["runtime"]["results_dir"]) / run_id
    output_dir.mkdir(parents=True)
    manifest = _create_manifest(config, data, validation, limit, run_id)
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    summaries = []
    ks = config["evaluation"]["k"]
    print(f"Run: {output_dir}", flush=True)

    try:
        for configured_model in models:
            started = time.monotonic()
            model = resolve_model(configured_model)
            model_status = {"resolved_config": model, "status": "running"}
            manifest["models"][model["name"]] = model_status
            write_json(manifest_path, manifest)
            print(f"Encoding {model['name']} ({model['revision']})", flush=True)

            # 1. 같은 모델로 문서와 질의를 임베딩해 서로 비교할 수 있는 벡터를 만든다.
            documents, queries, document_stats, query_stats = _encode_dataset(
                model, config["runtime"], data
            )

            # 2. 전체 문서에서 검색한 동일 순위를 자체 Recall 계산과 MTEB 대조에 함께 사용한다.
            rankings = exact_search(
                list(data.corpus), documents, list(data.queries), queries,
                top_k=max(100, max(ks)),
                batch_size=config["evaluation"]["query_batch_size"],
            )
            summary, rows, excluded = evaluate_recall(data.queries, data.qrels, rankings, ks)
            native = evaluate_with_mteb(data.qrels, rankings, ks, summary)

            # 3. 여러 모델 중 일부만 완료됐어도 완료한 모델의 결과는 즉시 저장한다.
            _save_model_results(output_dir, model, data, native, rows, excluded)
            summaries.append({
                "model": model["name"],
                "model_id": model["id"],
                "model_revision": model["revision"],
                "evaluated_queries": len(rows),
                "excluded_queries": len(excluded),
                **summary,
            })
            _write_summary(output_dir / "recall_summary.csv", summaries, ks)
            model_status.update(
                status="complete",
                document_stats=document_stats,
                query_stats=query_stats,
                evaluated_queries=len(rows),
                excluded_queries=excluded,
                elapsed_seconds=time.monotonic() - started,
            )
            write_json(manifest_path, manifest)
            print(model["name"], summary, flush=True)
        manifest["status"] = "complete"
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest.update(status=status, error=f"{type(error).__name__}: {error}")
        for model_status in manifest["models"].values():
            if model_status["status"] == "running":
                model_status.update(status=status, error=manifest["error"])
        raise
    finally:
        manifest["finished_at"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
        write_json(manifest_path, manifest)
    return output_dir


def probe_models(config, selected_models=None):
    """문서 2개·질의 2개로 실제 모델 추론이 가능한지 확인한다.

    검색 성능 실험이 아니므로 Recall 점수는 산출하지 않는다."""
    models = select_models(config, selected_models)
    data, _ = prepare_dataset(config)
    output_path = Path(config["runtime"]["results_dir"]) / "probes" / f"{_new_run_id()}.json"
    report = {
        "kind": "inference_probe",
        "benchmark_score": False,
        "status": "running",
        "models": {},
        "dataset_revision": data.metadata["revision"],
        "config": config,
    }
    write_json(output_path, report)
    try:
        for configured_model in models:
            started = time.monotonic()
            model_config = resolve_model(configured_model)
            model = EmbeddingModel(model_config, config["runtime"])
            try:
                documents, document_stats = model.encode(data.document_texts()[:2], "document")
                queries, query_stats = model.encode(list(data.queries.values())[:2], "query")
            finally:
                model.close()
            report["models"][model_config["name"]] = {
                "resolved_config": model_config,
                "document_shape": list(documents.shape),
                "query_shape": list(queries.shape),
                "document_stats": document_stats,
                "query_stats": query_stats,
                "elapsed_seconds": time.monotonic() - started,
            }
            write_json(output_path, report)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output_path, report)
    return output_path


def load_config(path):
    """YAML을 읽고 지원하지 않는 데이터 조건·잘못된 모델 이름·실행 크기를 미리 거부한다."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    dataset = config["dataset"]
    if dataset["split"] != "test" or dataset["input_template"] != "title_newline_text":
        raise ValueError("Only test/title_newline_text is supported")
    ks = config["evaluation"]["k"]
    if not ks or any(type(k) is not int or k < 1 for k in ks) or len(ks) != len(set(ks)):
        raise ValueError("K values must be distinct positive integers")
    if config["runtime"]["batch_size"] < 1 or config["evaluation"]["query_batch_size"] < 1:
        raise ValueError("Batch sizes must be positive")
    names = [model["name"] for model in config["models"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("Model names must be nonempty and unique")
    for model in config["models"]:
        name = model["name"]
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("Model name must be a safe filename")
        if model["max_length"] < 1:
            raise ValueError("max_length must be positive")
    return config


def main():
    parser = argparse.ArgumentParser(description="Legal embedding exact-retrieval benchmark")
    parser.add_argument("command", choices=("download", "prepare", "probe", "run"))
    parser.add_argument("--config", default="configs/legal_recall.yaml")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        if args.device:
            config["runtime"]["device"] = args.device
        if args.batch_size is not None:
            if args.batch_size < 1:
                raise ValueError("batch-size must be positive")
            config["runtime"]["batch_size"] = args.batch_size

        if args.command == "download":
            print(json.dumps(download_dataset(config["dataset"]), indent=2))
        elif args.command == "prepare":
            _, report = prepare_dataset(config)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        elif args.command == "probe":
            print(probe_models(config, args.models))
        else:
            print(run_benchmark(config, args.models, args.query_limit))
    except (ValueError, OSError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
